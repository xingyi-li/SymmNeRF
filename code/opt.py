import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--datadir", type=str, default='./data/SRN_Dataset/',
                        help='input data directory')
    parser.add_argument('--distributed', action='store_true',
                        help='if use distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='node rank for distributed training')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')

    parser.add_argument("--train_dataset", type=str, default='srns_dataset',
                        help='the training dataset')
    parser.add_argument('--mode', type=str, default='train',
                        help='train | val | test')
    parser.add_argument("--train_scene", type=str, default='cars',
                        help='optional, specify a subset of training scenes from training dataset')
    parser.add_argument('--eval_dataset', type=str, default='srns_dataset', help='the dataset to evaluate')
    parser.add_argument('--eval_scene', type=str, default='cars',
                        help='optional, specify a subset of scenes from eval_dataset to evaluate')

    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ckpt_path", type=str, default="",
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--no_load_opt", action='store_true',
                        help='do not load optimizer when reloading')
    parser.add_argument("--no_load_scheduler", action='store_true',
                        help='do not load scheduler when reloading')

    parser.add_argument("--N_iters", type=int, default=300000)

    parser.add_argument("--lrate_feature", type=float, default=1e-4, help='learning rate for feature extractor')
    parser.add_argument("--lrate_mlp", type=float, default=1e-4, help='learning rate for mlp')

    parser.add_argument("--lrate_decay_factor", type=float, default=0.5,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_steps", type=int, default=100000,
                        help='decay learning rate by a factor every specified number of steps')

    parser.add_argument("--src_view", type=str, default=None,
                        help='source view used to extract latent vector')

    parser.add_argument("--netdepth", type=int, default=3,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=3,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradN_randient step)')
    parser.add_argument("--chunk_size", type=int, default=1024 * 4,
                        help='number of rays processed in parallel, decrease if running out of memory')

    parser.add_argument("--batch_size", type=int, default=4, help='Object batch size')

    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64,
                        help='number of important samples per ray')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--no_bbox_step", type=int, default=100000,
                        help="Step to stop using bbox sampling")

    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--det", action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='apply the trick to avoid fitting to white background')

    parser.add_argument("--latent_dim", type=int, default=256,
                        help='dimension of latent vector')
    parser.add_argument("--hyper_num_hidden_layers", type=int, default=1,
                        help='number of hidden layers for hypernetwork')

    parser.add_argument("--i_print", type=int, default=10, help='frequency of terminal printout')
    parser.add_argument("--i_img", type=int, default=100, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')

    parser.add_argument("--use_spectral_norm", action='store_true',
                        help='if true, will use spectral norm in UNet depth network')

    parser.add_argument("--model", type=str, default='hypernerf_global',
                        help='which model to use')

    parser.add_argument("--eval_approx", action='store_true',
                        help='approximate evaluation (using only 1 random target view per object) '
                             'for use during development')

    parser.add_argument("--no_first_pool", action='store_true',
                        help='if true, skips first maxpool layer to avoid downscaling image '
                             'features too much (ResNet only)')
    parser.add_argument("--index_interp", type=str, default='bilinear',
                        help='')
    parser.add_argument("--index_padding", type=str, default='border',
                        help='')
    parser.add_argument("--upsample_interp", type=str, default='bilinear',
                        help='')
    parser.add_argument("--feature_scale", type=float, default=1.0)

    parser.add_argument("--num_local_layers", type=int, default=3)
    parser.add_argument("--local_feature_ch", type=int, default=512)

    parser.add_argument("--min_scale", type=float, default=0.25)
    parser.add_argument("--max_scale", type=float, default=1.0)
    parser.add_argument("--scale_anneal", type=float, default=0.0025)

    parser.add_argument('--specific_observation_idx', type=str, default=None,
                        help='only pick a subset of specific observations for each instance')

    parser.add_argument("--multicat", action="store_true",
                        help="Prepend category id to object id. Specify if model fits multiple categories.")
    return parser
