import sys
sys.path.append('../')
import torch
import torch.nn.functional as F
import numpy as np
from math import sqrt, exp

from model.nerf_helpers import get_rays_from_poses


class RaySampler(object):
    def __init__(self, data):
        super().__init__()
        self.instance_idx = data['index']
        self.render_imgs = data['images'] if 'images' in data.keys() else None
        self.render_poses = data['poses']
        # self.render_masks = data['masks']
        self.render_bboxes = data['bboxes']
        self.intrinsics = data['intrinsics'][:, 0]
        self.depth_range = data['depth_range'][0][0]

        self.src_view = data['src_view']
        # self.src_img = (data['src_image'][:, 0] - 0.5) / 0.5
        self.src_img = data['src_image'][:, 0]
        self.src_pose = data['src_pose'][:, 0]
        # self.src_mask = data['src_mask'][:, 0]
        # self.src_bbox = data['src_bbox'][:, 0]

        self.num_objs, self.num_views_per_obj, _, self.H, self.W = self.render_imgs.shape

        # self.target_idxs = list(range(self.num_views_per_obj))
        # self.target_idxs.remove(src_view)

        # if self.use_bbox and global_step >= args.no_bbox_step:
        #     self.use_bbox = False
        #     print("[Info] Stopped using bbox sampling @ iter", global_step)
        #
        # if not is_train or not self.use_bbox:
        #     all_bboxes = None

        all_rays_o = []
        all_rays_d = []
        for obj_idx in range(self.num_objs):
            rays_o, rays_d = get_rays_from_poses(self.H, self.W,
                                                 self.intrinsics[obj_idx], self.render_poses[obj_idx])
            rays_o = rays_o.reshape(-1, rays_o.shape[-1])
            rays_d = rays_d.reshape(-1, rays_d.shape[-1])
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)

        all_rays_o = torch.stack(all_rays_o)
        all_rays_d = torch.stack(all_rays_d)
        all_rgb_gt = self.render_imgs.permute(0, 1, 3, 4, 2).reshape(self.num_objs, -1, 3)

        self.rays_o = all_rays_o
        self.rays_d = all_rays_d
        self.rgb_gt = all_rgb_gt

    def bbox_sample(self, obj_idx, obj_views_ids, N_rand):
        image_ids = torch.from_numpy(np.random.choice(obj_views_ids, (N_rand,)))
        pix_bboxes = self.render_bboxes[obj_idx][image_ids]
        x = (torch.rand(N_rand) * (pix_bboxes[:, 2] + 1 - pix_bboxes[:, 0])
             + pix_bboxes[:, 0]).long()
        y = (torch.rand(N_rand) * (pix_bboxes[:, 3] + 1 - pix_bboxes[:, 1])
             + pix_bboxes[:, 1]).long()
        pix = torch.stack((image_ids, y, x), dim=-1)
        return pix

    # def get_all(self):
    #     ret = {'ray_o': self.rays_o.cuda(),
    #            'ray_d': self.rays_d.cuda(),
    #            'depth_range': self.depth_range.cuda(),
    #            'camera': self.camera.cuda(),
    #            'rgb': self.rgb.cuda() if self.rgb is not None else None,
    #            'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
    #            'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
    #     }
    #     return ret

    def get_all_single_image(self, index):
        select_inds = torch.arange(index * self.H * self.W, (index+1) * self.H * self.W)

        rays_o = self.rays_o[:1, select_inds]
        rays_d = self.rays_d[:1, select_inds]

        rgb = self.render_imgs[:1, index]
        pose = self.render_poses[:1, index]

        ret = {
            'instance_idx': self.instance_idx.long().cuda(),
            'rays_o': rays_o.cuda(),
            'rays_d': rays_d.cuda(),
            'pose': pose.cuda(),
            'intrinsics': self.intrinsics[:1].cuda(),
            'z_near': self.depth_range[0],
            'z_far': self.depth_range[1],
            'rgb': rgb.cuda(),
            'image_size': torch.tensor([self.H, self.W], dtype=torch.float32).cuda(),
            'src_img': self.src_img[:1].cuda(),
            'src_pose': self.src_pose[:1].cuda(),
            # 'src_mask': self.src_mask[:1].cuda(),
            # 'src_bbox': self.src_bbox[:1].cuda()
        }
        return ret

    def get_all(self):
        pass

    def sample_random_pixel(self, obj_idx, N_rand, use_bbox):
        if not use_bbox:
            self.render_bboxes = None
        obj_views_ids = range(self.num_views_per_obj)
        if self.render_bboxes is not None:
            pix = self.bbox_sample(obj_idx, obj_views_ids, N_rand)
            select_inds = pix[..., 0] * self.H * self.W + pix[..., 1] * self.W + pix[..., 2]
        else:
            select_inds = torch.randint(0, self.num_views_per_obj * self.H * self.W, (N_rand,))

        return select_inds

    def random_sample(self, N_rand, use_bbox=False):
        rays_o = []
        rays_d = []
        rgb = []
        for obj_idx in range(self.num_objs):
            select_inds = self.sample_random_pixel(obj_idx, N_rand, use_bbox)

            rays_o.append(self.rays_o[obj_idx][select_inds])
            rays_d.append(self.rays_d[obj_idx][select_inds])
            rgb.append(self.rgb_gt[obj_idx][select_inds])

        rays_o = torch.stack(rays_o)
        rays_d = torch.stack(rays_d)
        rgb = torch.stack(rgb)

        ret = {
            'instance_idx': self.instance_idx.long().cuda(),
            'rays_o': rays_o.cuda(),
            'rays_d': rays_d.cuda(),
            'intrinsics': self.intrinsics.cuda(),
            'z_near': self.depth_range[0],
            'z_far': self.depth_range[1],
            'rgb': rgb.cuda(),
            'image_size': torch.tensor([self.H, self.W], dtype=torch.float32).cuda(),
            'src_img': self.src_img.cuda(),
            'src_pose': self.src_pose.cuda(),
            # 'src_mask': self.src_mask.cuda(),
            # 'src_bbox': self.src_bbox.cuda(),
        }
        return ret
