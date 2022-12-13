import os, time
import torch
import numpy as np
import sys
sys.path.append('../')
from torch.utils.data import DataLoader
import lpips
from skimage import metrics
import imageio
import glob

import opt
from utils.general import *
from datasets import dataset_dict
from model import model_dict
from model.sample_ray import RaySampler
from model.render_ray import render_single_image


if __name__ == '__main__':
    parser = opt.config_parser()
    args = parser.parse_args()
    args.distributed = False

    device = "cuda:{}".format(args.local_rank)

    if args.eval_dataset == 'srns_dataset':
        logs_folder = os.path.join('../../logs', args.eval_dataset, args.eval_scene, args.expname)
    else:
        logs_folder = os.path.join('../../logs', args.eval_dataset, args.expname)
    eval_folder = os.path.join(logs_folder, 'eval_' + time.strftime("%m%d", time.localtime()) if not args.eval_approx else 'eval_approx')
    print('[Info] Saving results to {}'.format(eval_folder))
    os.makedirs(eval_folder, exist_ok=True)
    ckpts_folder = os.path.join(logs_folder, 'ckpts')

    model = model_dict[args.model](args, ckpts_folder)

    scene_name = args.eval_scene
    if args.eval_dataset == 'srns_dataset':
        out_scene_dir = os.path.join(eval_folder, '{}_{:06d}'.format(scene_name, model.start_step))
    else:
        out_scene_dir = os.path.join(eval_folder, '{:06d}'.format(model.start_step))
    os.makedirs(out_scene_dir, exist_ok=True)

    if args.eval_dataset == 'srns_dataset':
        test_dataset = dataset_dict[args.eval_dataset](args, 'test', scene=args.eval_scene)
    elif args.eval_dataset == 'dvr':
        test_dataset = dataset_dict[args.eval_dataset](args, 'test', list_prefix="softras_")
    else:
        test_dataset = dataset_dict[args.eval_dataset](args, 'test', list_prefix="gen_")
    # test_dataset = dataset_dict[args.eval_dataset](args, 'test', scene=args.eval_scene)

    test_loader = DataLoader(test_dataset, batch_size=1)

    save_prefix = scene_name

    total_num = 0
    # results_dict = {scene_name: {}} if args.multicat else {}
    results_dict = {scene_name: {}}
    cat_results_dict = {}
    sum_coarse_psnr = 0
    sum_fine_psnr = 0
    running_mean_coarse_psnr = 0
    running_mean_fine_psnr = 0
    sum_coarse_lpips = 0
    sum_fine_lpips = 0
    running_mean_coarse_lpips = 0
    running_mean_fine_lpips = 0
    sum_coarse_ssim = 0
    sum_fine_ssim = 0
    running_mean_coarse_ssim = 0
    running_mean_fine_ssim = 0

    lpips_vgg = lpips.LPIPS(net='vgg').to(device)

    if args.multicat:
        cat = None

    processed = sorted(glob(os.path.join(eval_folder, "*", "*")))[:-1]
    processed = [os.path.basename(i) for i in processed]
    for i, data in enumerate(test_loader):
        if os.path.basename(data['obj_dir'][0]) in processed:
            continue
        obj = os.path.basename(data['obj_dir'][0])
        if args.multicat:
            out_dir = os.path.join(out_scene_dir, data["category"][0], obj)

            if cat != data["category"][0]:
                if cat != None:
                    cat_results_dict[cat] = {'coarse_psnr': cat_sum_coarse_psnr / cat_total_num,
                                             'fine_psnr': cat_sum_fine_psnr / cat_total_num,
                                             'coarse_ssim': cat_sum_coarse_ssim / cat_total_num,
                                             'fine_ssim': cat_sum_fine_ssim / cat_total_num,
                                             'coarse_lpips': cat_sum_coarse_lpips / cat_total_num,
                                             'fine_lpips': cat_sum_fine_lpips / cat_total_num,
                                             }

                cat = data["category"][0]

                cat_sum_coarse_psnr = 0
                cat_sum_fine_psnr = 0
                cat_sum_coarse_ssim = 0
                cat_sum_fine_ssim = 0
                cat_sum_coarse_lpips = 0
                cat_sum_fine_lpips = 0

                cat_total_num = 0
        else:
            out_dir = os.path.join(out_scene_dir, obj)
        os.makedirs(out_dir, exist_ok=True)

        if args.noise_var > 0.0:
            w = torch.tensor([1., 0., 0.]).reshape(3, 1)
            noise = torch.randn_like(w) * args.noise_var
        else:
            noise = None

        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySampler(data)
            render_list = list(range(ray_sampler.render_imgs[0].shape[0]))
            render_list.remove(ray_sampler.src_view[0])

            if args.eval_approx:
                render_views = np.random.choice(render_list, 1)
            else:
                render_views = render_list
            # if args.eval_approx:
            #     render_views = np.random.choice(ray_sampler.render_imgs[0].shape[0], 1)
            # else:
            #     render_views = np.arange(ray_sampler.render_imgs[0].shape[0])

            for render_view in render_views:
                ray_batch = ray_sampler.get_all_single_image(render_view)
                file_id = os.path.basename(data['rgb_paths'][render_view][0]).split('.')[0]

                latent_vector = model.encode(ray_batch)

                ret = render_single_image(ray_sampler=ray_sampler,
                                          ray_batch=ray_batch,
                                          model=model,
                                          device=device,
                                          latent_vector=latent_vector,
                                          chunk_size=args.chunk_size,
                                          N_samples=args.N_samples,
                                          lindisp=args.lindisp,
                                          det=True,
                                          N_importance=args.N_importance,
                                          white_bkgd=args.white_bkgd,
                                          noise=noise)

                gt_rgb = ray_sampler.render_imgs[0][render_view].permute(1, 2, 0)
                coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
                coarse_err_map = torch.sum((coarse_pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
                coarse_err_map_colored = (colorize_np(coarse_err_map, range=(0., 1.)) * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(out_dir, '{}_err_map_coarse.png'.format(file_id)),
                                coarse_err_map_colored)
                coarse_pred_rgb_np = np.clip(coarse_pred_rgb.numpy(), a_min=0., a_max=1.)
                gt_rgb_np = gt_rgb.numpy()

                coarse_psnr = metrics.peak_signal_noise_ratio(coarse_pred_rgb_np, gt_rgb_np, data_range=1)
                coarse_ssim = metrics.structural_similarity(coarse_pred_rgb_np, gt_rgb_np, multichannel=True, data_range=1)
                coarse_lpips = lpips_vgg(coarse_pred_rgb[None, ...].permute(0, 3, 1, 2).float().to(device),
                                         gt_rgb[None, ...].permute(0, 3, 1, 2).float().to(device)).item()

                # Saving outputs ...
                src_rgb_np_uint8 = (255 * np.clip(ray_batch['src_img'][0].permute(1, 2, 0).detach().cpu().numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_dir, '{}_src_rgb.png'.format(file_id)), src_rgb_np_uint8)

                coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_dir, '{}_pred_coarse.png'.format(file_id)), coarse_pred_rgb)

                gt_rgb_np_uint8 = (255 * np.clip(gt_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_dir, '{}_gt_rgb.png'.format(file_id)), gt_rgb_np_uint8)

                coarse_pred_depth = ret['outputs_coarse']['depth'].detach().cpu()
                imageio.imwrite(os.path.join(out_dir, '{}_depth_coarse.png'.format(file_id)),
                                (coarse_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
                coarse_pred_depth_colored = colorize_np(coarse_pred_depth,
                                                        range=tuple(data['depth_range'][0][render_view].squeeze().cpu().numpy()))
                imageio.imwrite(os.path.join(out_dir, '{}_depth_vis_coarse.png'.format(file_id)),
                                (255 * coarse_pred_depth_colored).astype(np.uint8))
                coarse_acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()
                coarse_acc_map_colored = (colorize_np(coarse_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(out_dir, '{}_acc_map_coarse.png'.format(file_id)),
                                coarse_acc_map_colored)

                sum_coarse_psnr += coarse_psnr
                running_mean_coarse_psnr = sum_coarse_psnr / (i + 1) / len(render_views)
                sum_coarse_lpips += coarse_lpips
                running_mean_coarse_lpips = sum_coarse_lpips / (i + 1) / len(render_views)
                sum_coarse_ssim += coarse_ssim
                running_mean_coarse_ssim = sum_coarse_ssim / (i + 1) / len(render_views)

                if args.multicat:
                    cat_sum_coarse_psnr += coarse_psnr
                    cat_sum_coarse_lpips += coarse_lpips
                    cat_sum_coarse_ssim += coarse_ssim

                if ret['outputs_fine'] is not None:
                    fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
                    fine_pred_rgb_np = np.clip(fine_pred_rgb.numpy(), a_min=0., a_max=1.)

                    fine_psnr = metrics.peak_signal_noise_ratio(fine_pred_rgb_np, gt_rgb_np, data_range=1)
                    fine_ssim = metrics.structural_similarity(fine_pred_rgb_np, gt_rgb_np, multichannel=True,
                                                              data_range=1)
                    fine_lpips = lpips_vgg(fine_pred_rgb[None, ...].permute(0, 3, 1, 2).float().to(device),
                                           gt_rgb[None, ...].permute(0, 3, 1, 2).float().to(device)).item()

                    fine_err_map = torch.sum((fine_pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
                    fine_err_map_colored = (colorize_np(fine_err_map, range=(0., 1.)) * 255).astype(np.uint8)
                    imageio.imwrite(os.path.join(out_dir, '{}_err_map_fine.png'.format(file_id)),
                                    fine_err_map_colored)

                    fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                    imageio.imwrite(os.path.join(out_dir, '{}_pred_fine.png'.format(file_id)), fine_pred_rgb)
                    fine_pred_depth = ret['outputs_fine']['depth'].detach().cpu()
                    imageio.imwrite(os.path.join(out_dir, '{}_depth_fine.png'.format(file_id)),
                                    (fine_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
                    fine_pred_depth_colored = colorize_np(fine_pred_depth,
                                                          range=tuple(data['depth_range'][0][render_view].squeeze().cpu().numpy()))
                    imageio.imwrite(os.path.join(out_dir, '{}_depth_vis_fine.png'.format(file_id)),
                                    (255 * fine_pred_depth_colored).astype(np.uint8))
                    fine_acc_map = torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()
                    fine_acc_map_colored = (colorize_np(fine_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
                    imageio.imwrite(os.path.join(out_dir, '{}_acc_map_fine.png'.format(file_id)),
                                    fine_acc_map_colored)
                else:
                    fine_ssim = fine_lpips = fine_psnr = 0.

                sum_fine_psnr += fine_psnr
                running_mean_fine_psnr = sum_fine_psnr / (i + 1)
                sum_fine_lpips += fine_lpips
                running_mean_fine_lpips = sum_fine_lpips / (i + 1)
                sum_fine_ssim += fine_ssim
                running_mean_fine_ssim = sum_fine_ssim / (i + 1)

                if args.multicat:
                    cat_sum_fine_psnr += fine_psnr
                    cat_sum_fine_lpips += fine_lpips
                    cat_sum_fine_ssim += fine_ssim

                    print("=========================================================\n"
                          "category: {}, obj: {}, file_id: {} \n"
                          "current coarse psnr: {:03f}, current fine psnr: {:03f} \n"
                          "running mean coarse psnr: {:03f}, running mean fine psnr: {:03f} \n"
                          "current coarse ssim: {:03f}, current fine ssim: {:03f} \n"
                          "running mean coarse ssim: {:03f}, running mean fine ssim: {:03f} \n" 
                          "current coarse lpips: {:03f}, current fine lpips: {:03f} \n"
                          "running mean coarse lpips: {:03f}, running mean fine lpips: {:03f} \n"
                          "=========================================================\n"
                          .format(cat, os.path.basename(data['obj_dir'][0]), file_id,
                                  coarse_psnr, fine_psnr,
                                  running_mean_coarse_psnr, running_mean_fine_psnr,
                                  coarse_ssim, fine_ssim,
                                  running_mean_coarse_ssim, running_mean_fine_ssim,
                                  coarse_lpips, fine_lpips,
                                  running_mean_coarse_lpips, running_mean_fine_lpips))
                else:
                    print("=========================================================\n"
                          "scene_name: {}, obj: {}, file_id: {} \n"
                          "current coarse psnr: {:03f}, current fine psnr: {:03f} \n"
                          "running mean coarse psnr: {:03f}, running mean fine psnr: {:03f} \n"
                          "current coarse ssim: {:03f}, current fine ssim: {:03f} \n"
                          "running mean coarse ssim: {:03f}, running mean fine ssim: {:03f} \n"
                          "current coarse lpips: {:03f}, current fine lpips: {:03f} \n"
                          "running mean coarse lpips: {:03f}, running mean fine lpips: {:03f} \n"
                          "=========================================================\n"
                          .format(scene_name, os.path.basename(data['obj_dir'][0]), file_id,
                                  coarse_psnr, fine_psnr,
                                  running_mean_coarse_psnr, running_mean_fine_psnr,
                                  coarse_ssim, fine_ssim,
                                  running_mean_coarse_ssim, running_mean_fine_ssim,
                                  coarse_lpips, fine_lpips,
                                  running_mean_coarse_lpips, running_mean_fine_lpips))

                if args.multicat:
                    if obj not in results_dict.keys():
                        results_dict[obj] = {}
                    results_dict[obj][file_id] = {'coarse_psnr': coarse_psnr,
                                                  'fine_psnr': fine_psnr,
                                                  'coarse_ssim': coarse_ssim,
                                                  'fine_ssim': fine_ssim,
                                                  'coarse_lpips': coarse_lpips,
                                                  'fine_lpips': fine_lpips,
                                                  }
                else:
                    if obj not in results_dict[scene_name].keys():
                        results_dict[scene_name][obj] = {}
                    results_dict[scene_name][obj][file_id] = {'coarse_psnr': coarse_psnr,
                                                              'fine_psnr': fine_psnr,
                                                              'coarse_ssim': coarse_ssim,
                                                              'fine_ssim': fine_ssim,
                                                              'coarse_lpips': coarse_lpips,
                                                              'fine_lpips': fine_lpips,
                                                              }

                total_num += 1
                if args.multicat:
                    cat_total_num += 1

    mean_coarse_psnr = sum_coarse_psnr / total_num
    mean_fine_psnr = sum_fine_psnr / total_num
    mean_coarse_lpips = sum_coarse_lpips / total_num
    mean_fine_lpips = sum_fine_lpips / total_num
    mean_coarse_ssim = sum_coarse_ssim / total_num
    mean_fine_ssim = sum_fine_ssim / total_num

    if args.multicat:
        cat_results_dict[cat] = {'coarse_psnr': cat_sum_coarse_psnr / cat_total_num,
                                 'fine_psnr': cat_sum_fine_psnr / cat_total_num,
                                 'coarse_ssim': cat_sum_coarse_ssim / cat_total_num,
                                 'fine_ssim': cat_sum_fine_ssim / cat_total_num,
                                 'coarse_lpips': cat_sum_coarse_lpips / cat_total_num,
                                 'fine_lpips': cat_sum_fine_lpips / cat_total_num,
                                 }

        print('-------ALL-------\n'
              'final coarse psnr: {}, final fine psnr: {}\n'
              'final coarse ssim: {}, final fine ssim: {} \n'
              'final coarse lpips: {}, final fine lpips: {} \n'
              .format(mean_coarse_psnr, mean_fine_psnr,
                      mean_coarse_ssim, mean_fine_ssim,
                      mean_coarse_lpips, mean_fine_lpips))

        results_dict['coarse_mean_psnr'] = mean_coarse_psnr
        results_dict['fine_mean_psnr'] = mean_fine_psnr
        results_dict['coarse_mean_ssim'] = mean_coarse_ssim
        results_dict['fine_mean_ssim'] = mean_fine_ssim
        results_dict['coarse_mean_lpips'] = mean_coarse_lpips
        results_dict['fine_mean_lpips'] = mean_fine_lpips

        for cat in cat_results_dict.keys():
            print('-------{}-------\n'
                  'final coarse psnr: {}, final fine psnr: {}\n'
                  'final coarse ssim: {}, final fine ssim: {} \n'
                  'final coarse lpips: {}, final fine lpips: {} \n'
                  .format(cat, cat_results_dict[cat]['coarse_psnr'], cat_results_dict[cat]['fine_psnr'],
                          cat_results_dict[cat]['coarse_ssim'], cat_results_dict[cat]['fine_ssim'],
                          cat_results_dict[cat]['coarse_lpips'], cat_results_dict[cat]['fine_lpips']))

        results_dict['cat_metrics'] = cat_results_dict

    else:
        print('-------{}-------\n'
              'final coarse psnr: {}, final fine psnr: {}\n'
              'final coarse ssim: {}, final fine ssim: {} \n'
              'final coarse lpips: {}, final fine lpips: {} \n'
              .format(scene_name, mean_coarse_psnr, mean_fine_psnr,
                      mean_coarse_ssim, mean_fine_ssim,
                      mean_coarse_lpips, mean_fine_lpips))

        results_dict[scene_name]['coarse_mean_psnr'] = mean_coarse_psnr
        results_dict[scene_name]['fine_mean_psnr'] = mean_fine_psnr
        results_dict[scene_name]['coarse_mean_ssim'] = mean_coarse_ssim
        results_dict[scene_name]['fine_mean_ssim'] = mean_fine_ssim
        results_dict[scene_name]['coarse_mean_lpips'] = mean_coarse_lpips
        results_dict[scene_name]['fine_mean_lpips'] = mean_fine_lpips

    if args.eval_dataset == 'srns_dataset':
        f = open("{}/metrics_{}_{}.txt".format(eval_folder, save_prefix, model.start_step), "w")
    else:
        f = open("{}/metrics_{}.txt".format(eval_folder, model.start_step), "w")
    f.write(str(results_dict))
    f.close()
