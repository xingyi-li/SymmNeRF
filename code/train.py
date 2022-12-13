import time
import imageio
import shutil
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import random

import opt
from utils.general import *
from datasets import dataset_dict, create_training_dataset
from model import model_dict
from model.sample_ray import RaySampler
from model.render_ray import render_rays, log_view_to_tb


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def train(args):
    device = "cuda:{}".format(args.local_rank)

    if args.train_dataset == 'srns_dataset':
        logs_folder = os.path.join('../logs', args.train_dataset, args.train_scene, args.expname)
    else:
        logs_folder = os.path.join('../logs', args.train_dataset, args.expname)
    print('[Info] Outputs will be saved to {}'.format(logs_folder))
    os.makedirs(logs_folder, exist_ok=True)

    ckpts_folder = os.path.join(logs_folder, 'ckpts')
    os.makedirs(ckpts_folder, exist_ok=True)

    visuals_folder = os.path.join(logs_folder, 'visuals')
    os.makedirs(os.path.join(visuals_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(visuals_folder, 'val'), exist_ok=True)

    # Save the args and config files
    f = os.path.join(logs_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        f = os.path.join(logs_folder, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # Create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              worker_init_fn=lambda _: np.random.seed(),
                              num_workers=args.workers,
                              pin_memory=True,
                              sampler=train_sampler,
                              shuffle=True if train_sampler is None else False)

    if args.eval_dataset == 'srns_dataset':
        val_dataset = dataset_dict[args.eval_dataset](args, 'val', scene=args.eval_scene)
    elif args.eval_dataset == 'dvr':
        val_dataset = dataset_dict[args.eval_dataset](args, 'val', list_prefix="softras_")
    else:
        val_dataset = dataset_dict[args.eval_dataset](args, 'val', list_prefix="gen_")

    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    # Create model
    model = model_dict[args.model](args, ckpts_folder)

    # Create criterion
    criterion = torch.nn.MSELoss()

    tb_dir = os.path.join(logs_folder, 'tensorboard')
    if args.local_rank == 0:
        writer = SummaryWriter(tb_dir)
        print('[Info] Saving tensorboard files to {}'.format(tb_dir))

    scalars_to_log = {}
    global_step = model.start_step + 1
    epoch = 0

    # Main loop
    while global_step < args.N_iters + 1:
        np.random.seed()
        for train_data in train_loader:
            time0 = time.time()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop
            # Load training rays
            ray_sampler = RaySampler(train_data)
            ray_batch = ray_sampler.random_sample(args.N_rand, use_bbox=(global_step <= args.no_bbox_step))

            latent_vector = model.encode(ray_batch)

            ret = render_rays(ray_batch=ray_batch,
                              model=model,
                              device=device,
                              latent_vector=latent_vector,
                              N_samples=args.N_samples,
                              lindisp=args.lindisp,
                              N_importance=args.N_importance,
                              det=args.det,
                              raw_noise_std=args.raw_noise_std,
                              white_bkgd=args.white_bkgd)

            # Compute loss
            model.optimizer.zero_grad()
            loss = criterion(ret['outputs_coarse']['rgb'], ray_batch['rgb'])

            if ret['outputs_fine'] is not None:
                fine_loss = criterion(ret['outputs_fine']['rgb'], ray_batch['rgb'])
                loss += fine_loss

            loss.backward()
            scalars_to_log['loss'] = loss.item()
            model.optimizer.step()
            # model.scheduler.step()
            if hasattr(model, 'warmup_scheduler'):
                model.scheduler.step(model.scheduler.last_epoch + 1)
                model.warmup_scheduler.dampen()
            else:
                model.scheduler.step()


            # scalars_to_log['lr'] = model.scheduler.get_last_lr()[0]
            scalars_to_log['lr'] = model.optimizer.param_groups[0]['lr']

            # End of core optimization loop
            dt = time.time() - time0

            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    # Write mse and psnr stats
                    mse_error = img2mse(ret['outputs_coarse']['rgb'], ray_batch['rgb']).item()
                    scalars_to_log['train/coarse-loss'] = mse_error
                    scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(mse_error)
                    if ret['outputs_fine'] is not None:
                        mse_error = img2mse(ret['outputs_fine']['rgb'], ray_batch['rgb']).item()
                        scalars_to_log['train/fine-loss'] = mse_error
                        scalars_to_log['train/fine-psnr-training-batch'] = mse2psnr(mse_error)

                    logstr = '{} Epoch: {} step: [{}/{}] '.format(args.expname, epoch, global_step, args.N_iters)
                    for k in scalars_to_log.keys():
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)
                    logstr += ' time: {:.05f} seconds'.format(dt)
                    print(logstr)

                if global_step % args.i_weights == 0:
                    print('[Info] Saving checkpoints at {} to {}...'.format(global_step, ckpts_folder))
                    fpath = os.path.join(ckpts_folder, 'model_{:06d}.pth'.format(global_step))
                    model.save_model(fpath)

                if global_step % args.i_img == 0:
                    print('[Info] Logging a random validation view...')
                    val_data = next(val_loader_iterator)

                    tmp_ray_sampler = RaySampler(val_data)

                    render_list = list(range(tmp_ray_sampler.render_imgs[0].shape[0]))
                    render_list.remove(tmp_ray_sampler.src_view[0])
                    render_view = np.random.choice(render_list, 1)[0]
                    # render_view = np.random.choice(tmp_ray_sampler.render_imgs[0].shape[0], 1)[0]
                    gt_img = tmp_ray_sampler.render_imgs[0][render_view].permute(1, 2, 0)
                    rgb_im, depth_im, acc_map = log_view_to_tb(writer, global_step, args, model, device,
                                                               tmp_ray_sampler, render_view,
                                                               gt_img, prefix='val/')
                    if ret['outputs_fine'] is not None:
                        imageio.imwrite(os.path.join(visuals_folder, 'val',
                                                     'rgb_src-gt-coarse-fine_{}.png'.format(global_step)),
                                        to8b(np.array(rgb_im.permute(1, 2, 0))))
                        imageio.imwrite(os.path.join(visuals_folder, 'val',
                                                     'depth-coarse-fine_{}.png'.format(global_step)),
                                        to8b(np.array(depth_im.permute(1, 2, 0))))
                        imageio.imwrite(os.path.join(visuals_folder, 'val',
                                                     'acc-coarse-fine_{}.png'.format(global_step)),
                                        to8b(np.array(acc_map.permute(1, 2, 0))))
                    else:
                        imageio.imwrite(os.path.join(visuals_folder, 'val',
                                                     'rgb_src-gt-coarse_{}.png'.format(global_step)),
                                        to8b(np.array(rgb_im.permute(1, 2, 0))))
                        imageio.imwrite(os.path.join(visuals_folder, 'val',
                                                     'depth-coarse_{}.png'.format(global_step)),
                                        to8b(np.array(depth_im.permute(1, 2, 0))))
                        imageio.imwrite(os.path.join(visuals_folder, 'val',
                                                     'acc-coarse_{}.png'.format(global_step)),
                                        to8b(np.array(acc_map.permute(1, 2, 0))))
                    torch.cuda.empty_cache()

                    print('[Info] Logging current training view...')

                    tmp_ray_train_sampler = RaySampler(train_data)

                    render_list = list(range(tmp_ray_train_sampler.render_imgs[0].shape[0]))
                    render_list.remove(tmp_ray_train_sampler.src_view[0])
                    render_view = np.random.choice(render_list, 1)[0]
                    # render_view = np.random.choice(tmp_ray_train_sampler.render_imgs[0].shape[0], 1)[0]
                    gt_img = tmp_ray_train_sampler.render_imgs[0][render_view].permute(1, 2, 0)
                    rgb_im, depth_im, acc_map = log_view_to_tb(writer, global_step, args, model, device,
                                                               tmp_ray_train_sampler, render_view,
                                                               gt_img, prefix='train/')
                    if ret['outputs_fine'] is not None:
                        imageio.imwrite(os.path.join(visuals_folder, 'train',
                                                     'rgb_src-gt-coarse-fine_{}.png'.format(global_step)),
                                        to8b(np.array(rgb_im.permute(1, 2, 0))))
                        imageio.imwrite(os.path.join(visuals_folder, 'train',
                                                     'depth-coarse-fine_{}.png'.format(global_step)),
                                        to8b(np.array(depth_im.permute(1, 2, 0))))
                        imageio.imwrite(os.path.join(visuals_folder, 'train',
                                                     'acc-coarse-fine_{}.png'.format(global_step)),
                                        to8b(np.array(acc_map.permute(1, 2, 0))))
                    else:
                        imageio.imwrite(os.path.join(visuals_folder, 'train',
                                                     'rgb_src-gt-coarse_{}.png'.format(global_step)),
                                        to8b(np.array(rgb_im.permute(1, 2, 0))))
                        imageio.imwrite(os.path.join(visuals_folder, 'train',
                                                     'depth-coarse_{}.png'.format(global_step)),
                                        to8b(np.array(depth_im.permute(1, 2, 0))))
                        imageio.imwrite(os.path.join(visuals_folder, 'train',
                                                     'acc-coarse_{}.png'.format(global_step)),
                                        to8b(np.array(acc_map.permute(1, 2, 0))))
                    torch.cuda.empty_cache()

            global_step += 1
            if global_step == args.no_bbox_step:
                print("[Info] Stop using bbox sampling @ iter", global_step)
            if global_step > model.start_step + args.N_iters + 1:
                break
        epoch += 1


if __name__ == '__main__':
    parser = opt.config_parser()
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    train(args)
