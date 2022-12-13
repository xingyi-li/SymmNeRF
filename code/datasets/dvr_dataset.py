import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from utils.data_utils import get_image_to_tensor, get_mask_to_tensor


class DVRDataset(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 mode,
                 list_prefix="softras_",
                 image_size=None,
                 scale_focal=True,
                 max_imgs=100000,
                 z_near=1.2,
                 z_far=4.0):
        super().__init__()
        self.args = args
        self.mode = mode
        self.base_dir = args.datadir

        use_source_lut = True if args.src_view is not None else False
        if use_source_lut:
            print("Using views from list", args.src_view)
            with open(args.src_view, "r") as f:
                tmp = [x.strip().split() for x in f.readlines()]
            self.source_lut = {
                x[0] + "/" + x[1]: torch.tensor(list(map(int, x[2:])), dtype=torch.long)
                for x in tmp
            }

        categories = [x for x in glob.glob(os.path.join(self.base_dir, "*")) if os.path.isdir(x)]
        obj_lists = [os.path.join(x, list_prefix + '{}.lst'.format(mode)) for x in categories]

        all_objs = []
        for obj_list in obj_lists:
            if not os.path.exists(obj_list):
                continue
            category_dir = os.path.dirname(obj_list)
            category = os.path.basename(category_dir)
            with open(obj_list, "r") as f:
                objs = [(category, os.path.join(category_dir, x.strip())) for x in f.readlines()]
            all_objs.extend(objs)

        self.all_objs = all_objs

        self.image_to_tensor = get_image_to_tensor()
        self.mask_to_tensor = get_mask_to_tensor()
        print("[Info] Loading DVR dataset: {}, mode: {}, "
              "type: {}, {} objects in total".format(self.base_dir,
                                                     mode,
                                                     "ShapeNet",
                                                     len(self.all_objs)))

        self.image_size = image_size
        self._coord_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False

    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        category, obj_dir = self.all_objs[index]

        rgb_paths = [x for x in glob.glob(os.path.join(obj_dir, "image", "*"))
                     if x.endswith(".jpg") or x.endswith(".png")]
        rgb_paths = sorted(rgb_paths)
        mask_paths = sorted(glob.glob(os.path.join(obj_dir, "mask", "*.png")))
        if len(mask_paths) == 0:
            mask_paths = [None] * len(rgb_paths)

        if len(rgb_paths) <= self.max_imgs:
            sel_indices = np.arange(len(rgb_paths))
        else:
            sel_indices = np.random.choice(len(rgb_paths), self.max_imgs, replace=False)
            rgb_paths = [rgb_paths[i] for i in sel_indices]
            mask_paths = [mask_paths[i] for i in sel_indices]

        cam_path = os.path.join(obj_dir, "cameras.npz")
        all_cam = np.load(cam_path)

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        focal = None

        for idx, (rgb_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
            i = sel_indices[idx]
            img = imageio.imread(rgb_path)[..., :3]
            if self.scale_focal:
                x_scale = img.shape[1] / 2.0
                y_scale = img.shape[0] / 2.0
                xy_delta = 1.0
            else:
                x_scale = y_scale = 1.0
                xy_delta = 0.0

            if mask_path is not None:
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 2:
                    mask = mask[..., None]
                mask = mask[..., :1]

            # ShapeNet
            wmat_inv_key = "world_mat_inv_" + str(i)
            wmat_key = "world_mat_" + str(i)
            if wmat_inv_key in all_cam:
                extr_inv_mtx = all_cam[wmat_inv_key]
            else:
                extr_inv_mtx = all_cam[wmat_key]
                if extr_inv_mtx.shape[0] == 3:
                    extr_inv_mtx = np.vstack((extr_inv_mtx, np.array([0, 0, 0, 1])))
                extr_inv_mtx = np.linalg.inv(extr_inv_mtx)

            intr_mtx = all_cam["camera_mat_" + str(i)]
            fx, fy = intr_mtx[0, 0], intr_mtx[1, 1]
            assert abs(fx - fy) < 1e-9
            fx = fx * x_scale
            if focal is None:
                focal = fx
            else:
                assert abs(fx - focal) < 1e-5
            pose = extr_inv_mtx

            pose = torch.tensor(pose, dtype=torch.float32) @ self._coord_trans

            img_tensor = self.image_to_tensor(img)
            if mask_path is not None:
                mask_tensor = self.mask_to_tensor(mask)

                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                rnz = np.where(rows)[0]
                cnz = np.where(cols)[0]
                if len(rnz) == 0:
                    raise RuntimeError("ERROR: Bad image at", rgb_path, "please investigate!")
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
                bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
                all_masks.append(mask_tensor)
                all_bboxes.append(bbox)

            all_imgs.append(img_tensor)
            all_poses.append(pose)

        if mask_path is not None:
            all_bboxes = torch.stack(all_bboxes)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        if len(all_masks) > 0:
            all_masks = torch.stack(all_masks)
        else:
            all_masks = None

        if self.image_size is not None and all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale

            if mask_path is not None:
                all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            if all_masks is not None:
                all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        H, W = all_imgs.shape[-2:]

        if self.args.src_view is None:
            src_view = np.random.choice(len(rgb_paths), 1)[0]
        else:
            # src_view = int(self.args.src_view)
            obj_id = category + "/" + os.path.basename(obj_dir)
            src_view = self.source_lut[obj_id]

        src_img = all_imgs[src_view:src_view+1]
        src_pose = all_poses[src_view:src_view+1]
        # src_mask = all_masks[src_view:src_view+1]
        # src_bbox = all_bboxes[src_view:src_view+1]
        src_path = rgb_paths[src_view]

        K = np.array([[focal, 0., W // 2, 0.],
                      [0., focal, H // 2, 0],
                      [0., 0, 1, 0],
                      [0, 0, 0, 1]]).astype(np.float32)
        intrinsics = torch.from_numpy(K).view(1, 4, 4).repeat(len(all_imgs), 1, 1)
        depth_range = torch.tensor([self.z_near, self.z_far], dtype=torch.float32).view(1, -1).repeat(len(all_imgs), 1)

        ret = {
            "obj_dir": obj_dir,
            "category": category,
            "index": index,
            "intrinsics": intrinsics,
            "rgb_paths": rgb_paths,
            "images": all_imgs,
            "poses": all_poses,
            "depth_range": depth_range,
            "src_view": src_view,
            "src_path": src_path,
            "src_image": src_img,
            # "src_mask": src_mask,
            "src_pose": src_pose,
            # "src_bbox": src_bbox
        }
        if all_masks is not None:
            ret["masks"] = all_masks

        ret["bboxes"] = all_bboxes
        return ret
