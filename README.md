# SymmNeRF: Learning to Explore Symmetry Prior for Single-View View Synthesis (ACCV 2022)
[Xingyi Li](https://xingyi-li.github.io/)<sup>1</sup>,
[Chaoyi Hong](https://www.semanticscholar.org/author/Chaoyi-Hong/2047434854)<sup>1</sup>,
[Yiran Wang](https://scholar.google.com/citations?user=p_RnaI8AAAAJ&hl)<sup>1</sup>,
[Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>,
[Ke Xian](https://sites.google.com/site/kexian1991/)<sup>2*</sup>,
[Guosheng Lin](https://guosheng.github.io/)<sup>2</sup>

<sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Nanyang Technological University

### [Paper](https://github.com/xingyi-li/SymmNeRF/blob/main/pdf/symmnerf-paper.pdf) | [arXiv](https://arxiv.org/abs/2209.14819) | [Video](https://youtu.be/YWIjScmMWwc) | [Supp](https://github.com/xingyi-li/SymmNeRF/blob/main/pdf/symmnerf-supp.pdf) | [Poster](https://github.com/xingyi-li/SymmNeRF/blob/main/pdf/symmnerf-poster.pdf) 

This repository is the official PyTorch implementation of the ACCV 2022 paper "SymmNeRF: Learning to Explore Symmetry Prior for Single-View View Synthesis".

## Installation
```
conda create -n symmnerf python=3.8
conda activate symmnerf
conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install configargparse imageio opencv-python scipy tensorboard lpips scikit-image tqdm pytorch_warmup
```

## Data Preparation
Please refer to [pixel-nerf](https://github.com/sxyu/pixel-nerf#getting-the-data) and download the datasets including SRN chair/car (128x128) and the ShapeNet 64x64 dataset from NMR. 

## Training

### ShapeNet Single-Category (SRN)
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12346 train.py --config configs/srns.txt --expname srns_cars --model hypernerf_symm_local --N_rand 256 --N_importance 64 --local_feature_ch 1024 --N_iters 500000 --distributed --num_local_layers 2 --train_scene cars --eval_scene cars

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12347 train.py --config configs/srns.txt --expname srns_chairs --model hypernerf_symm_local --N_rand 256 --N_importance 64 --local_feature_ch 1024 --N_iters 500000 --distributed --num_local_layers 2 --train_scene chairs --eval_scene chairs
```

### ShapeNet Multiple Categories (NMR)
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12348 train.py --config configs/dvr.txt --expname sn64 --model hypernerf_symm_local --N_rand 256 --N_importance 0 --local_feature_ch 1024 --N_iters 500000 --distributed --num_local_layers 2 --no_first_pool

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12349 train.py --config configs/dvr_gen.txt --expname sn64_unseen --model hypernerf_symm_local --N_rand 256 --N_importance 0 --local_feature_ch 1024 --N_iters 500000 --distributed --num_local_layers 2 --no_first_pool
```

## Testing

### ShapeNet Single-Category (SRN)
```
CUDA_VISIBLE_DEVICES=0 python eval.py --config ../configs/srns.txt --expname srns_cars --model hypernerf_symm_local --src_view 64 --local_feature_ch 1024 --num_local_layers 2 --train_scene cars --eval_scene cars

CUDA_VISIBLE_DEVICES=0 python eval.py --config ../configs/srns.txt --expname srns_chairs --model hypernerf_symm_local --src_view 64 --local_feature_ch 1024 --num_local_layers 2 --train_scene chairs --eval_scene chairs
```

### ShapeNet Multiple Categories (NMR)
```
CUDA_VISIBLE_DEVICES=0 python eval.py --config ../configs/dvr.txt --expname sn64 --model hypernerf_symm_local --local_feature_ch 1024 --num_local_layers 2 --src_view ../viewlist/src_dvr.txt --no_first_pool --N_importance 0 --multicat

CUDA_VISIBLE_DEVICES=3 python eval.py --config ../configs/dvr_gen.txt --expname sn64_unseen --model hypernerf_symm_local --local_feature_ch 1024 --num_local_layers 2 --src_view ../viewlist/src_gen.txt --no_first_pool --N_importance 0 --multicat
```


## Citation
If you find our work useful in your research, please consider to cite our paper:

```
@InProceedings{li2022symmnerf,
    author    = {Li, Xingyi and Hong, Chaoyi and Wang, Yiran and Cao, Zhiguo and Xian, Ke and Lin, Guosheng},
    title     = {SymmNeRF: Learning to Explore Symmetry Prior for Single-View View Synthesis},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2022},
    pages     = {1726-1742}
}
```

## Acknowledgements
This code borrows heavily from [pixel-nerf](https://github.com/sxyu/pixel-nerf#getting-the-data). Part of the code is based on [IBRNet](https://github.com/googleinterns/IBRNet). We thank the respective authors for open sourcing their methods. 
