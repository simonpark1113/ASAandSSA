import numpy as np
import torch
from torch import nn
from typing import Union, List, Tuple, Type

from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder

class symmetricalAttention(nn.Module):
    def __init__(self, in_channels, mid_channel=32,eps=1e-7 ):
        super(symmetricalAttention, self).__init__()
        self.eps = eps
        # implement spatial attention
        # self.reduce_channel_x0 = nn.Conv3d(in_channels, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        # self.reduce_channel_x_flip = nn.Conv3d(in_channels, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.reduce_channel = nn.Conv3d(in_channels, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_0, x_flip):
        # print(x_0.shape, x_flip.shape)
        residual = x_0
        
        x_0_projected = self.reduce_channel(x_0)
        x_0_projected = self.relu(x_0_projected)
        
        x_flip_projected = self.reduce_channel(x_flip)
        x_flip_projected = self.relu(x_flip_projected)
        
        numer = torch.sum(x_0_projected * x_flip_projected, dim=1)
        denom = torch.sqrt(torch.sum(x_0_projected ** 2, dim=1)) * \
            torch.sqrt(torch.sum(x_flip_projected ** 2, dim=1)) + self.eps
        # print(numer.shape, denom.shape)
        sym_attn = numer / denom
        # print(sym_attn.shape)
        # Bigger the difference -> more attention
        sym_attn = 1.0-sym_attn
        sym_attn = sym_attn.unsqueeze(1)
        # print(x_max.shape, x_avg.shape, sym_attn.shape)
        spatial_attn = self.conv(sym_attn)
        spatial_attn = self.sigmoid(spatial_attn)
        x_0 = x_0 * spatial_attn
        # print("final: ", x_0.shape)
        # assert False
        return x_0+residual

class synthsegAttentionModule(nn.Module):
    def __init__(self, in_channels, mid_channel):
        super(synthsegAttentionModule, self).__init__()
        self.project_initial = nn.Conv3d(in_channels, mid_channel, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.InstanceNorm3d(mid_channel, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.project_logits = nn.Conv3d(mid_channel, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, synthseg):
        residual = x
        input_cat = torch.cat([x, synthseg], dim=1)
        output = self.project_initial(input_cat)
        output = self.batch_norm(output)
        output = self.relu(output)
        output = self.project_logits(output)
        output = self.sigmoid(output)
        output = output*x
        return output + residual

# class synthsegAttentionModule(nn.Module):
#     def __init__(self, in_channels, mid_channel, out_channel):
#         super(synthsegAttentionModule, self).__init__()
#         self.project_initial = nn.Conv3d(in_channels, mid_channel, kernel_size=3, stride=1, padding=1)
#         self.batch_norm = nn.BatchNorm3d(mid_channel)
#         self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.1)
#         self.merge_conv = nn.Conv3d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1,)
#         self.batch_norm2 = nn.BatchNorm3d(out_channel)
#         self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.1)

        
#     def forward(self, x, synthseg):
#         residual = x
#         input_cat = torch.cat([x, synthseg], dim=1)
#         projected_output = self.project_initial(input_cat)
#         projected_output = self.batch_norm(projected_output)
#         projected_output = self.relu(projected_output)
#         projected_output = self.merge_conv(projected_output)
#         projected_output = self.batch_norm2(projected_output)
#         projected_output = self.relu2(projected_output)
#         return projected_output + residual

class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        print("decoder for customUnetBoth Att built!!")
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs


        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        spatial_attn_layers = []
        synth_attn_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            ))
            spatial_attn_layers.append(symmetricalAttention(input_features_skip))
            synth_attn_layers.append(synthsegAttentionModule(input_features_skip*2, input_features_skip))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
            

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.synth_attn_layers = nn.ModuleList(synth_attn_layers)
        self.spatial_attn_layers = nn.ModuleList(spatial_attn_layers)

    def forward(self, skips, skips_flip, skips_label_0, skips_label_flip):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            skip_i_0 = self.synth_attn_layers[s](skips[-(s+2)], skips_label_0[-(s+2)])
            skip_i_flip = self.synth_attn_layers[s](skips_flip[-(s+2)], skips_label_flip[-(s+2)])
            skip_i = self.spatial_attn_layers[s](skip_i_0, skip_i_flip)
            x = torch.cat((x, skip_i), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output