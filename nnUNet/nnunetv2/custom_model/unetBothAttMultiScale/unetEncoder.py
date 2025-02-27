import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op

import torch.nn.functional as F

class CAMSA_block(nn.Module):
    def __init__(self,
                input_channels: int,
                n_scales: int = 2,
                min_channels: int = 16
                ):
        super().__init__()
        self.n_scales = n_scales
        self.min_channels = min_channels

        # self.alpha_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(n_scales)])
        # self.beta_list = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(n_scales)])

        self.multi_scale_convs = nn.ModuleList([])
        # self.mssa_convs = nn.ModuleList([])
        self.c_recovery_convs = nn.ModuleList([])
        for i in range(n_scales):
            kernel_size = 3 + 2*i
            padding_size = 1+i
            dilation = 1+i
            internal_channels = max(min_channels, int(input_channels//2**i))
            # print(input_channels, input_channels/2**i, kernel_size, padding_size)
            # mult_scale_conv_i = nn.Sequential(
            #     nn.Conv3d(in_channels= input_channels, out_channels= int(input_channels/2**i), kernel_size=kernel_size, padding=padding_size, bias=False),
            #     nn.InstanceNorm3d(int(input_channels/2**i)),
            #     nn.LeakyReLU(inplace=True)
            #     nn.Conv3d(in_channels= int(input_channels/2**i), out_channels= int(input_channels/2**i), kernel_size=1, padding=0, bias=False),
            #     nn.InstanceNorm3d(int(input_channels/2**i)),
            #     nn.LeakyReLU(inplace=True)
            # )
            mult_scale_conv_i = nn.Sequential(
                nn.Conv3d(in_channels= input_channels, out_channels= internal_channels, kernel_size=3, padding=padding_size, dilation=dilation, bias=False),
                nn.BatchNorm3d(internal_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(in_channels= internal_channels, out_channels= internal_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm3d(internal_channels),
                nn.LeakyReLU(inplace=True)
            )
            self.multi_scale_convs.append(mult_scale_conv_i)

            # spatial_attn = nn.Sequential(
            #     nn.Conv3d(int(input_channels/2**i), 1, kernel_size=1, padding=0, bias=False),
            #     nn.Sigmoid()
            # )
            # self.mssa_convs.append(spatial_attn)
            recov_conv_i = nn.Sequential(
                nn.Conv3d(internal_channels, input_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(input_channels),
                nn.LeakyReLU(inplace=True)
            )
            self.c_recovery_convs.append(recov_conv_i)



        pass

    def forward(self, x):
        feature_aggregation = 0
        # print(f"input shape: {x.shape}")
        for scale in range(self.n_scales):
            feature = F.avg_pool3d(x, kernel_size=2**scale, stride=2**scale, padding=0)
            feature = self.multi_scale_convs[scale](feature)
            ############################## removed due to NAN loss ##############################
            # spatial_attn_map = self.mssa_convs[scale](feature)
            # feature = feature * (1.0 - spatial_attn_map) * self.alpha_list[scale] + feature * spatial_attn_map * self.beta_list[scale]
            #####################################################################################
            # print(f"scale: {scale}, feature shape: {feature.shape}")
            feature = F.interpolate(self.c_recovery_convs[scale](feature), size=None, scale_factor=2**scale, mode='trilinear')
            # print(f"scale: {scale}, feature shape after interpolation: {feature.shape}")
            feature_aggregation += feature
        feature_aggregation /= self.n_scales

        return feature_aggregation+x

# class MultiscaleBlock(nn.Module):
#     def __init__(self, input_channels: int, n_scales: int = 2, min_channels: int = 16):
#         super().__init__()
#         self.n_scales = n_scales
#         self.min_channels = min_channels

#         self.multi_scale_convs = nn.ModuleList([])
        
#         for i in range(n_scales):
#             kernel_size = 3 + 2*i
#             padding_size = kernel_size // 2
#             mult_scale_conv_i = nn.Conv3d(in_channels= input_channels, out_channels= int(input_channels/2**i), kernel_size=kernel_size, padding=padding_size)
#             self.multi_scale_convs.append(mult_scale_conv_i)

#             spatial_attn = nn.Sequential(
#                 nn.Conv3d(int(input_channels/2**i), 1, kernel_size=1, padding=0, bias=False),
#                 nn.Sigmoid()
#             )
#             self.mssa_convs.append(spatial_attn)

#             self.c_recovery_convs.append(nn.Conv3d(int(input_channels/2**i), input_channels, kernel_size=3, padding=1, bias=False))

#     def forward(self, x):
#         feature_aggregation = 0
#         for scale in range(self.n_scales):
#             feature = F.avg_pool3d(x, kernel_size=2**scale, stride=2**scale, padding=0)
#             feature = self.multi_scale_convs[scale](feature)
#             spatial_attn_map = self.mssa_convs[scale](feature)
#             feature = feature * (1.0 - spatial_attn_map) * self.alpha_list[scale] + feature * spatial_attn_map * self.beta_list[scale]
#             feature = F.interpolate(self.c_recovery_convs[scale](feature), size=None, scale_factor=2**scale, mode='trilinear')
#             feature_aggregation += feature
#         feature_aggregation /= self.n_scales

#         return feature_aggregation+x

class PlainConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv'
                 ):

        print("Successfully modified encoder loading!!!!!!!!!!!!!!!!!!!!!")
        print("version without spatial attention")
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(StackedConvBlocks(
                n_conv_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], conv_stride,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ))
            stage_modules.append(CAMSA_block(features_per_stage[s], n_scales=3, min_channels=16))
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output

