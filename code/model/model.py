import sys
sys.path.append('../')
import os
import torch
import pytorch_warmup as warmup

from model.hypernetwork import HyperNetworkSymmLocal
from model.nerf_helpers import get_embedder
from model.feature_network import create_resnet_symm_local


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model


class HyperNeRFResNetSymmLocal(object):
    def __init__(self, args, ckpts_folder):
        super().__init__()
        self.args = args
        device = torch.device('cuda:{}'.format(args.local_rank))
        embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.embed_fn = embed_fn
        self.embeddirs_fn = embeddirs_fn

        self.hypernetwork = HyperNetworkSymmLocal(hyper_in_ch=args.latent_dim,
                                                  hyper_num_hidden_layers=args.hyper_num_hidden_layers,
                                                  hyper_hidden_ch=args.latent_dim,
                                                  hidden_ch=args.netwidth,
                                                  num_hidden_layers=args.netdepth,
                                                  num_local_layers=args.num_local_layers,
                                                  input_ch=self.input_ch,
                                                  input_ch_views=self.input_ch_views,
                                                  local_feature_ch=args.local_feature_ch,
                                                  outermost_linear=True).to(device)

        if args.N_importance > 0:
            self.hypernetwork_fine = HyperNetworkSymmLocal(hyper_in_ch=args.latent_dim,
                                                           hyper_num_hidden_layers=args.hyper_num_hidden_layers,
                                                           hyper_hidden_ch=args.latent_dim,
                                                           hidden_ch=args.netwidth,
                                                           num_hidden_layers=args.netdepth,
                                                           num_local_layers=args.num_local_layers,
                                                           input_ch=self.input_ch,
                                                           input_ch_views=self.input_ch_views,
                                                           local_feature_ch=args.local_feature_ch,
                                                           outermost_linear=True).to(device)
        else:
            self.hypernetwork_fine = None

        self.feature_net = create_resnet_symm_local(arch='resnet34', latent_dim=args.latent_dim,
                                                    pretrained=True, index_interp=args.index_interp,
                                                    index_padding=args.index_padding,
                                                    upsample_interp=args.upsample_interp,
                                                    feature_scale=args.feature_scale,
                                                    use_first_pool=not args.no_first_pool).to(device)

        # Optimizer and learning rate scheduler
        if self.hypernetwork_fine is not None:
            self.optimizer = torch.optim.AdamW([
                {'params': self.hypernetwork.parameters()},
                {'params': self.hypernetwork_fine.parameters()},
                {'params': self.feature_net.parameters(), 'lr': args.lrate_feature}],
                lr=args.lrate_mlp)
        else:
            self.optimizer = torch.optim.AdamW([
                {'params': self.hypernetwork.parameters()},
                {'params': self.feature_net.parameters(), 'lr': args.lrate_feature}],
                lr=args.lrate_mlp)

        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
        #                                                  step_size=args.lrate_decay_steps,
        #                                                  gamma=args.lrate_decay_factor)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=args.N_iters)
        self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

        self.start_step = self.load_from_ckpt(ckpts_folder,
                                              load_opt=not args.no_load_opt,
                                              load_scheduler=not args.no_load_scheduler)

        if args.distributed:
            self.hypernetwork = torch.nn.parallel.DistributedDataParallel(
                self.hypernetwork,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

            self.feature_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.feature_net)
            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

            if self.hypernetwork_fine is not None:
                self.hypernetwork_fine = torch.nn.parallel.DistributedDataParallel(
                    self.hypernetwork_fine,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank
                )

    def encode(self, ray_batch):
        latent_vector = self.feature_net(ray_batch['src_img'])
        return latent_vector

    def switch_to_eval(self):
        self.hypernetwork.eval()
        self.feature_net.eval()
        if self.hypernetwork_fine is not None:
            self.hypernetwork_fine.eval()

    def switch_to_train(self):
        self.hypernetwork.train()
        self.feature_net.train()
        if self.hypernetwork_fine is not None:
            self.hypernetwork_fine.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'warmup_scheduler': self.warmup_scheduler.state_dict(),
                   'hypernetwork': de_parallel(self.hypernetwork).state_dict(),
                   'feature_net': de_parallel(self.feature_net).state_dict()
                   }

        if self.hypernetwork_fine is not None:
            to_save['hypernetwork_fine'] = de_parallel(self.hypernetwork_fine).state_dict()

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])
            self.warmup_scheduler.load_state_dict(to_load['warmup_scheduler'])

        self.hypernetwork.load_state_dict(to_load['hypernetwork'])
        self.feature_net.load_state_dict(to_load['feature_net'])

        if self.hypernetwork_fine is not None and 'hypernetwork_fine' in to_load.keys():
            self.hypernetwork_fine.load_state_dict(to_load['hypernetwork_fine'])

    def load_from_ckpt(self, ckpts_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        """
        Load model from existing checkpoints and return the current step.

        :param ckpts_folder: the directory that stores ckpts
        :param load_opt:
        :param load_scheduler:
        :param force_latest_ckpt:
        :return: the current starting step
        """

        # All existing ckpts
        ckpts = []
        if os.path.exists(ckpts_folder):
            ckpts = [os.path.join(ckpts_folder, f)
                     for f in sorted(os.listdir(ckpts_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print('[Info] Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('[Info] No ckpts found, training from scratch...')
            step = 0

        return step
