import sys
sys.path.append('../')
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
import imageio
import numpy as np
from utils.data_utils import get_image_to_tensor, get_mask_to_tensor
from utils.general import parse_comma_separated_integers, pick


class SRNsDataset(Dataset):
    def __init__(self,
                 args,
                 mode,
                 scene='cars',
                 image_size=(128, 128),
                 world_scale=1.0,
                 specific_observation_idx=None):  # For few-shot case: Can pick specific observations only
        super().__init__()
        self.args = args
        assert scene in ['cars', 'chairs'], 'Only support cars & chairs for SRNsDataset'
        self.folder_path = os.path.join(args.datadir, scene + '_{}'.format(mode))
        if scene == 'chairs' and mode == 'train':
            self.folder_path = os.path.join(self.folder_path, 'chairs_2.0_train')
        print("[Info] Loading SRNs dataset: {}".format(self.folder_path))
        self.mode = mode
        self.scene = scene
        self.all_intrinsics_files = sorted(glob.glob(os.path.join(self.folder_path, "*", "intrinsics.txt")))
        self.all_obj = sorted(glob.glob(os.path.join(self.folder_path, "*")))
        self.image_to_tensor = get_image_to_tensor()
        self.mask_to_tensor = get_mask_to_tensor()
        self.image_size = image_size
        self.world_scale = world_scale
        if specific_observation_idx is not None:
            self.specific_observation_idx = parse_comma_separated_integers(specific_observation_idx)
        else:
            self.specific_observation_idx = None
        self._coord_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))

        if scene == 'chairs':
            self.z_near = 1.25
            self.z_far = 2.75
        else:
            self.z_near = 0.8
            self.z_far = 1.8

    def __len__(self):
        return len(self.all_intrinsics_files)

    def __getitem__(self, index):
        intrinsics_file = self.all_intrinsics_files[index]
        obj_dir = os.path.dirname(intrinsics_file)
        rgb_paths = sorted(glob.glob(os.path.join(obj_dir, "rgb", "*")))
        pose_paths = sorted(glob.glob(os.path.join(obj_dir, "pose", "*")))
        assert len(rgb_paths) == len(pose_paths)

        if self.specific_observation_idx is not None:
            rgb_paths = pick(rgb_paths, self.specific_observation_idx)
            pose_paths = pick(pose_paths, self.specific_observation_idx)

        with open(intrinsics_file, "r") as intrinsics:
            lines = intrinsics.readlines()
            focal, cx, cy, _ = map(float, lines[0].split())
            H, W = map(int, lines[-1].split())

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        for rgb_path, pose_path in zip(rgb_paths, pose_paths):
            img = imageio.imread(rgb_path)[..., :3]
            img_tensor = self.image_to_tensor(img)
            mask = (img != 255).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)

            pose = torch.from_numpy(np.loadtxt(pose_path, dtype=np.float32).reshape(4, 4))
            pose = pose @ self._coord_trans

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            # if len(rnz) == 0:
            #     raise RuntimeError("ERROR: Bad image at", rgb_path, "please investigate!")
            try:
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
            except:
                bbox = torch.tensor([0, 0, 0, 0], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        if (H, W) != self.image_size:
            scale = self.image_size[0] / H
            focal *= scale
            cx *= scale
            cy *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        # H, W = self.image_size

        if self.world_scale != 1.0:
            focal *= self.world_scale
            all_poses[:, :3, 3] *= self.world_scale

        if self.args.src_view is None:
            src_view = np.random.choice(len(rgb_paths), 1)[0]
        else:
            src_view = int(self.args.src_view)

        if self.specific_observation_idx is not None:
            src_img = all_imgs
            src_pose = all_poses
            src_path = rgb_paths[0]
        else:
            src_img = all_imgs[src_view:src_view+1]
            src_pose = all_poses[src_view:src_view+1]
            # src_mask = all_masks[src_view:src_view+1]
            # src_bbox = all_bboxes[src_view:src_view+1]
            src_path = rgb_paths[src_view]

        # if self.mode != 'train':
        #     inds = list(range(len(rgb_paths)))
        #     inds.remove(src_view)
        #     all_imgs = all_imgs[inds]
        #     all_masks = all_masks[inds]
        #     all_bboxes = all_bboxes[inds]
        #     all_poses = all_poses[inds]

        # inds = list(range(len(rgb_paths)))
        # inds.remove(src_view)
        # all_imgs = all_imgs[inds]
        # all_masks = all_masks[inds]
        # all_bboxes = all_bboxes[inds]
        # all_poses = all_poses[inds]

        # src_path = rgb_paths[src_view]
        # rgb_paths.remove(src_path)

        K = np.array([[focal, 0., cx, 0.],
                      [0., focal, cy, 0],
                      [0., 0, 1, 0],
                      [0, 0, 0, 1]]).astype(np.float32)
        intrinsics = torch.from_numpy(K).view(1, 4, 4).repeat(len(all_imgs), 1, 1)
        depth_range = torch.tensor([self.z_near, self.z_far], dtype=torch.float32).view(1, -1).repeat(len(all_imgs), 1)

        ret = {
            "obj_dir": obj_dir,
            "index": index,
            "intrinsics": intrinsics,
            "rgb_paths": rgb_paths,
            "images": all_imgs,
            "masks": all_masks,
            "bboxes": all_bboxes,
            "poses": all_poses,
            "depth_range": depth_range,
            "src_view": src_view,
            "src_path": src_path,
            "src_image": src_img,
            # "src_mask": src_mask,
            "src_pose": src_pose,
            # "src_bbox": src_bbox
        }

        return ret
