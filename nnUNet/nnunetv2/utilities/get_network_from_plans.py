import pydoc
import warnings
from typing import Union

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.custom_model.unet.customUnet import customUnet
from nnunetv2.custom_model.unetSym.customUnetSym import customUnetSym
from nnunetv2.custom_model.attnUnet.attnUnet import build_attnUnet_default
from nnunetv2.custom_model.unetSymOnlyProject.customUnetSymOnlyProject import customUnetSymOnlyProject
from nnunetv2.custom_model.unetSynthSeg.customUnetSynthSeg import customUnetSynthSeg
from nnunetv2.custom_model.unetSymSynthSegEarly.customUnetSymSynthsegEarly import customUnetSymSynthSegEarly
from nnunetv2.custom_model.unetSynthsegAtt.customUnetSynthsegAtt import customUnetSynthsegOnlyAtt
from nnunetv2.custom_model.unetBothAtt.customUnetBothAtt import customUnetBothAtt
from nnunetv2.custom_model.unetBothAttMultiScale.customUnetBothAttMS import customUnetBothAttMS


def get_network_from_plans(arch_class_name, arch_kwargs, arch_kwargs_req_import, input_channels, output_channels,
                           allow_init=True, deep_supervision: Union[bool, None] = None):
    network_class = arch_class_name
    architecture_kwargs = dict(**arch_kwargs)
    for ri in arch_kwargs_req_import:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    if deep_supervision is not None:
        architecture_kwargs['deep_supervision'] = deep_supervision

    if(network_class == "customUnet"):
        print("succesfully modified network_class to customUnet")
        nw_class = customUnet
        network = nw_class(
            input_channels=input_channels,
            num_classes=output_channels,
            **architecture_kwargs
        )
    elif(network_class == "customUnetSymm"):
        print("succesfully modified network_class to customUnetSymm")
        nw_class = customUnetSym
        network = nw_class(
            input_channels=input_channels//2,
            num_classes=output_channels,
            **architecture_kwargs
        )
    elif(network_class=='customUnetSymOnlyProject'):
        print("succesfully modified network_class to customUnetSymOnlyProject")
        nw_class = customUnetSymOnlyProject
        network = nw_class(
            input_channels=input_channels//2,
            num_classes=output_channels,
            **architecture_kwargs
        )
    elif(network_class == "attnUnet"):
        print("succesfully modified network_class to attnUnet")
        network = build_attnUnet_default()
    elif(network_class == "customUnetSynthSeg"):
        print("succesfully modified network_class to customUnetSynthSeg")
        nw_class = customUnetSynthSeg
        network = nw_class(
            input_channels=20,
            num_classes=output_channels,
            **architecture_kwargs
        )
    elif(network_class == "customUnetSymSynthSegEarly"):
        print("succesfully modified network_class to customUnetSymSynthSegEarly")
        nw_class = customUnetSymSynthSegEarly
        network = nw_class(
            input_channels=20,
            num_classes=output_channels,
            **architecture_kwargs
        )
    elif(network_class == "customUnetSynthsegAtt"):
        print("succesfully modified network_class to customUnetSynthsegAtt")
        nw_class = customUnetSynthsegOnlyAtt
        network = nw_class(
            input_channels=input_channels//2,
            num_classes=output_channels,
            **architecture_kwargs
        )
    elif(network_class == "customUnetBothAtt"):
        print("succesfully modified network_class to customUnetBothAtt")
        nw_class = customUnetBothAtt
        network = nw_class(
            input_channels=1,
            num_classes=output_channels,
            **architecture_kwargs
        )
    elif(network_class == "customUnetBothAttMS"):
        print("succesfully modified network_class to customUnetBothAttMS")
        nw_class = customUnetBothAttMS
        network = nw_class(
            input_channels=1,
            num_classes=output_channels,
            **architecture_kwargs
        )
    else:
        nw_class = pydoc.locate(network_class)
        # sometimes things move around, this makes it so that we can at least recover some of that
        if nw_class is None:
            warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                        f'dynamic_network_architectures.architectures...')
            import dynamic_network_architectures
            nw_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                                network_class.split(".")[-1],
                                                'dynamic_network_architectures.architectures')
            if nw_class is not None:
                print(f'FOUND IT: {nw_class}')
            else:
                raise ImportError('Network class could not be found, please check/correct your plans file')
        network = nw_class(
            input_channels=input_channels,
            num_classes=output_channels,
            **architecture_kwargs
        )
    

    

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    
    return network

if __name__ == "__main__":
    import torch

    model = get_network_from_plans(
        arch_class_name="dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
        arch_kwargs={
            "n_stages": 7,
            "features_per_stage": [32, 64, 128, 256, 512, 512, 512],
            "conv_op": "torch.nn.modules.conv.Conv2d",
            "kernel_sizes": [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
            "strides": [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm2d",
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
        input_channels=1,
        output_channels=4,
        allow_init=True,
        deep_supervision=True,
    )
    data = torch.rand((8, 1, 256, 256))
    target = torch.rand(size=(8, 1, 256, 256))
    outputs = model(data) # this should be a list of torch.Tensor