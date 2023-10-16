# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from mmdet.models.necks.bfp import  BFP


from ..builder import NECKS
from torchvision.ops import DeformConv2d


@NECKS.register_module()
class SSDNeck(BaseModule):
    """Extra layers of SSD backbone to generate multi-scale feature maps.

    Args:
        in_channels (Sequence[int]): Number of input channels per scale.
        out_channels (Sequence[int]): Number of output channels per scale.
        level_strides (Sequence[int]): Stride of 3x3 conv per level.
        level_paddings (Sequence[int]): Padding size of 3x3 conv per level.
        l2_norm_scale (float|None): L2 normalization layer init scale.
            If None, not use L2 normalization on the first input feature.
        last_kernel_size (int): Kernel size of the last conv layer.
            Default: 3.
        use_depthwise (bool): Whether to use DepthwiseSeparableConv.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 level_strides,
                 level_paddings,
                 l2_norm_scale=20.,
                 last_channel=256,
                 first_channel=1024,
                 last_kernel_size=3,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=[
                     dict(
                         type='Xavier', distribution='uniform',
                         layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d'),
                 ]):
        super(SSDNeck, self).__init__(init_cfg)

        print('neck_len' ,len(in_channels) ,len(out_channels),len(level_strides)  )

        assert len(out_channels) > len(in_channels)
        assert len(out_channels) - len(in_channels) == len(level_strides)
        assert len(level_strides) == len(level_paddings)
        assert in_channels == out_channels[:len(in_channels)]

        if l2_norm_scale:
            self.l2_norm = L2Norm(in_channels[0], l2_norm_scale)
            self.init_cfg += [
                dict(
                    type='Constant',
                    val=self.l2_norm.scale,
                    override=dict(name='l2_norm'))
            ]

        self.extra_layers = nn.ModuleList()
        extra_layer_channels = out_channels[len(in_channels):]
        second_conv = DepthwiseSeparableConvModule if \
            use_depthwise else ConvModule

        for i, (out_channel, stride, padding) in enumerate(
                zip(extra_layer_channels, level_strides, level_paddings)):
            kernel_size = last_kernel_size \
                if i == len(extra_layer_channels) - 1 else 3
            per_lvl_convs = nn.Sequential(
                ConvModule(
                    out_channels[len(in_channels) - 1 + i],
                    out_channel // 2,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                second_conv(
                    out_channel // 2,
                    out_channel,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            self.extra_layers.append(per_lvl_convs)

        ##############################
        self.first_channel = first_channel
        self.my_offset= nn.Conv2d(self.first_channel,18,3,padding=1,bias=False)
        self.my_dcn2d = DeformConv2d(self.first_channel,self.first_channel,3,stride=1,padding=1)


        self.last_channel = last_channel
        self.conv1024 = nn.Conv2d(1024, self.last_channel, 1)
        self.conv512 = nn.Conv2d(512, self.last_channel, 1)
        self.conv256 = nn.Conv2d(256, self.last_channel, 1)

    def my_amend(self, type,inputs):

        if type == "none":
            output = inputs
        elif type == 'dcn_backbone':
            offset = self.my_offset(inputs)
            output=self.my_dcn2d(inputs,offset)
        else:
                output = inputs

        return output


    def get256channel(self, inputlist):

         # 测试 gfl 归一成512
        for i in range(len(inputlist)):
            if (inputlist[i]).shape[1] ==self.last_channel :
                pass
            else:
                if (inputlist[i]).shape[1] ==256 :
                    temp256 = self.conv256(inputlist[i])
                    inputlist[i] = temp256
                elif (inputlist[i]).shape[1] ==1024:
                   temp1024 = self.conv1024(inputlist[i])
                   inputlist[i] = temp1024
                elif (inputlist[i]).shape[1] ==512:
                   temp512 = self.conv512(inputlist[i])
                   inputlist[i] = temp512

        return inputlist



    # input   = 512,64,64
           #    1024，32，32

    # output1 = 512，16，16
    # output2 = 256，8，8
    # output3 = 512，4，4
    # output4 = 512，2，2
    # output5 = 512，1，1

    def forward(self, inputs):
        """Forward function."""
        outs = [feat for feat in inputs]

        if hasattr(self, 'l2_norm'):
            outs[0] = self.l2_norm(outs[0])
        feat = outs[-1]
        outs[-1] = self.my_amend('dcn_backbone', outs[-1])  # my2_DCN

        return tuple(outs)


    # SSD  original
    # def forward(self, inputs):
    #     """Forward function."""
    #     outs = [feat for feat in inputs]
    #     if hasattr(self, 'l2_norm'):
    #         outs[0] = self.l2_norm(outs[0])
    #
    #     feat = outs[-1]
    #     for layer in self.extra_layers:
    #         feat = layer(feat)
    #         outs.append(feat)
    #     return tuple(outs)

class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        """L2 normalization layer.

        Args:
            n_dims (int): Number of dimensions to be normalized
            scale (float, optional): Defaults to 20..
            eps (float, optional): Used to avoid division by zero.
                Defaults to 1e-10.
        """
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        """Forward function."""
        # normalization layer convert to FP32 in FP16 training
        x_float = x.float()
        norm = x_float.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return (self.weight[None, :, None, None].float().expand_as(x_float) *
                x_float / norm).type_as(x)
