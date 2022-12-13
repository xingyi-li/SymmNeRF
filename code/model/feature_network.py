import torch
import torch.nn as nn
import torch.nn.functional as F


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetSymmLocal(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, index_interp='bilinear',
                 index_padding='border', upsample_interp='bilinear', feature_scale=1.0, use_first_pool=True):
        super().__init__()
        # feature_scale factor to scale all latent by. Useful (<1) if image is extremely large, to fit in memory.
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp

        self.register_buffer("latent", torch.empty(1, 1, 1, 1), persistent=False)
        self.register_buffer("latent_scaling", torch.empty(2, dtype=torch.float32), persistent=False)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        if self.feature_scale != 1.0:
            x = F.interpolate(x, scale_factor=self.feature_scale,
                              mode='bilinear' if self.feature_scale > 1.0 else 'area',
                              align_corners=True if self.feature_scale > 1.0 else False,
                              recompute_scale_factor=True)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        latents = [x]
        if self.use_first_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        latents.append(x)

        x = self.layer2(x)
        latents.append(x)

        x = self.layer3(x)
        latents.append(x)

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        align_corners = False if self.index_interp == 'nearest' else True
        latent_sz = latents[0].shape[-2:]

        for i in range(len(latents)):
            latents[i] = F.interpolate(latents[i], latent_sz,
                                       mode=self.upsample_interp,
                                       align_corners=align_corners)
        self.latent = torch.cat(latents, dim=1)
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def index(self, xyz, src_poses, intrinsics, image_size, noise=None):
        xyz = torch.reshape(xyz, [xyz.shape[0], -1, xyz.shape[-1]])
        # Models in ShapeNet have already been processed so that in their canonical poses,
        # the Y-Z plane is the plane of the reflection symmetry

        M = torch.tensor([[-1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]], dtype=torch.float32).to(xyz.device)

        xyz_symm = (M @ xyz.unsqueeze(-1))[..., 0]

        focal = intrinsics[:, 0, 0].view(-1, 1, 1).repeat(1, xyz.shape[-2], 2)
        focal[..., 1] = focal[..., 1] * -1
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        c = torch.stack([cx, cy], -1).unsqueeze(1).repeat(1, xyz.shape[-2], 1)

        rot = src_poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, src_poses[:, :3, 3:])  # (B, 3, 1)
        w2c = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)
        # Transform query points into the camera space of the source view
        xyz_rot = torch.matmul(w2c[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
        xyz = xyz_rot + w2c[:, None, :3, 3]
        xyz_symm_rot = torch.matmul(w2c[:, None, :3, :3], xyz_symm.unsqueeze(-1))[..., 0]
        xyz_symm = xyz_symm_rot + w2c[:, None, :3, 3]

        uv = -xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
        uv = uv * focal + c

        uv_symm = -xyz_symm[:, :, :2] / xyz_symm[:, :, 2:]  # (SB, B, 2)
        uv_symm = uv_symm * focal + c

        scale = self.latent_scaling / image_size
        uv = uv * scale - 1.0
        uv_symm = uv_symm * scale - 1.0

        uv = uv.unsqueeze(2)  # (B, N, 1, 2)
        uv_symm = uv_symm.unsqueeze(2)  # (B, N, 1, 2)

        samples = F.grid_sample(self.latent, uv, align_corners=True,
                                mode=self.index_interp, padding_mode=self.index_padding)
        local_feature = samples[:, :, :, 0].transpose(1, 2)

        samples_symm = F.grid_sample(self.latent, uv_symm, align_corners=True,
                                     mode=self.index_interp, padding_mode=self.index_padding)
        local_feature_symm = samples_symm[:, :, :, 0].transpose(1, 2)

        local_feature = torch.cat([local_feature, local_feature_symm], -1)

        return local_feature


def _resnet_symm_local(arch, latent_dim, block, layers, pretrained, progress, **kwargs):
    model = ResNetSymmLocal(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch],
                                                        progress=progress)
        model.load_state_dict(state_dict)
    model.fc = nn.Linear(512 * block.expansion, latent_dim)
    return model


def create_resnet_symm_local(arch='resnet50', latent_dim=256, pretrained=False, progress=True, **kwargs):
    if arch == 'resnet18':
        return _resnet_symm_local('resnet18', latent_dim, BasicBlock, [2, 2, 2, 2], pretrained, progress,
                       **kwargs)
    elif arch == 'resnet34':
        return _resnet_symm_local('resnet34', latent_dim, BasicBlock, [3, 4, 6, 3], pretrained, progress,
                       **kwargs)
    elif arch == 'resnet50':
        return _resnet_symm_local('resnet50', latent_dim, Bottleneck, [3, 4, 6, 3], pretrained, progress,
                       **kwargs)
    elif arch == 'resnet101':
        return _resnet_symm_local('resnet101', latent_dim, Bottleneck, [3, 4, 23, 3], pretrained, progress,
                       **kwargs)
    elif arch == 'resnet152':
        return _resnet_symm_local('resnet152', latent_dim, Bottleneck, [3, 8, 36, 3], pretrained, progress,
                       **kwargs)
    elif arch == 'resnext50_32x4d':
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 4
        return _resnet_symm_local('resnext50_32x4d', latent_dim, Bottleneck, [3, 4, 6, 3],
                       pretrained, progress, **kwargs)
    elif arch == 'resnext101_32x8d':
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 8
        return _resnet_symm_local('resnext101_32x8d', latent_dim, Bottleneck, [3, 4, 23, 3],
                       pretrained, progress, **kwargs)
    elif arch == 'wide_resnet50_2':
        kwargs['width_per_group'] = 64 * 2
        return _resnet_symm_local('wide_resnet50_2', latent_dim, Bottleneck, [3, 4, 6, 3],
                       pretrained, progress, **kwargs)
    elif arch == 'wide_resnet101_2':
        kwargs['width_per_group'] = 64 * 2
        return _resnet_symm_local('wide_resnet101_2', latent_dim, Bottleneck, [3, 4, 23, 3],
                       pretrained, progress, **kwargs)
    else:
        raise NotImplementedError
