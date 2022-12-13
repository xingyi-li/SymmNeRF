import sys
sys.path.append('../')
import torch
from torch.utils.data import Dataset

from datasets.srns_dataset import SRNsDataset
from datasets.dvr_dataset import DVRDataset


dataset_dict = {
    'srns_dataset': SRNsDataset,
    'dvr': DVRDataset,
    'dvr_gen': DVRDataset
}


def create_training_dataset(args):
    print('[Info] Training dataset: {}'.format(args.train_dataset))
    mode = 'train'

    if args.train_dataset == 'srns_dataset':
        train_dataset = dataset_dict[args.train_dataset](args, mode, scene=args.train_scene)
    elif args.train_dataset == 'dvr':
        train_dataset = dataset_dict[args.train_dataset](args, mode, list_prefix="softras_")
    else:
        train_dataset = dataset_dict[args.train_dataset](args, mode, list_prefix="gen_")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None

    return train_dataset, train_sampler
