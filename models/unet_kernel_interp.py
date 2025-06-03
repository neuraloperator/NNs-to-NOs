import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import Unet
import copy
from utilities import resize, bilinear_interpolate

class InterpConv2d(nn.Conv2d):
    def __init__(self,
                scale_dict,          # dict containing scale
                in_channels: int,
                out_channels: int,
                kernel_size,
                stride = 1,
                padding = 0,
                dilation = 1,
                groups: int = 1,
                bias: bool = True,
                padding_mode: str = 'zeros',
                interpolation_mode: str = 'bilinear',
                device=None,
                dtype=None):
        
        super(InterpConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, padding_mode, device, dtype)
        
        self.scale_dict = scale_dict
        self.interpolation_mode = interpolation_mode
        self.base_padding = padding

        if interpolation_mode == 'bilinear':
            self.interpolate = bilinear_interpolate
        elif interpolation_mode == 'fourier':
            self.interpolate = resize
        else:
            raise NotImplementedError("Interpolation method not implemented.")

    def interpolate_kernel(self, scale, weight):
        if scale == 1.0:
            return weight, self.base_padding

        # Upsample the convolutional kernel weights
        old_shape = weight.shape
        if isinstance(self.kernel_size, tuple):
            kernel_size = tuple(int(scale * i - scale + 1) for i in self.kernel_size)
            new_shape = (old_shape[0], old_shape[1], kernel_size[0], kernel_size[1])
        else:                                                           # it is int
            kernel_size = int(scale * self.kernel_size - scale + 1) # new kernel size
            new_shape = (old_shape[0], old_shape[1], kernel_size, kernel_size)

        new_kernel = self.interpolate(weight, (new_shape[2], new_shape[3]))

        normalization = (new_shape[2] * new_shape[3]) / (old_shape[2] * old_shape[3])
        new_kernel /=  normalization                            # normalize by scale

        if isinstance(self.base_padding, tuple):
            padding = tuple(int(i * scale) for i in self.base_padding)
        elif isinstance(self.base_padding, str):
            padding = self.base_padding
        else:                                       # it is int
            padding = int(self.base_padding * scale)

        return new_kernel, padding
    
    def forward(self, input):
        new_kernel, new_padding = self.interpolate_kernel(self.scale_dict["scale"], self.weight)
        self.padding = new_padding
        return self._conv_forward(input, new_kernel, self.bias)

class InOutInterpolateConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self,
                scale_dict,          # dict containing scale
                in_channels, 
                out_channels, 
                kernel_size, 
                stride=1, 
                padding=0, 
                output_padding=0, 
                groups=1, 
                bias=True, 
                dilation=1, 
                padding_mode='zeros', 
                interpolation_mode: str = 'bilinear',
                device=None, 
                dtype=None):
        
        super(InOutInterpolateConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, 
                                           stride, padding, output_padding, groups, 
                                           bias, dilation, padding_mode, device, dtype)
        
        self.scale_dict = scale_dict
        self.interpolation_mode = interpolation_mode
        self.base_padding = padding
        self.base_output_padding = output_padding

        if interpolation_mode == 'bilinear':
            self.interpolate = bilinear_interpolate
        elif interpolation_mode == 'fourier':
            self.interpolate = resize
        else:
            raise NotImplementedError("Interpolation method not implemented.")

    # Adapted from PDE Arena U-Net code
    def match_dims(self, x1, target_size):
        diffY = target_size[-2] - x1.size()[-2]
        diffX = target_size[-1] - x1.size()[-1]

        return F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    
    def forward(self, input, output_size=None):
        scale = self.scale_dict["scale"]

        # Interpolate input
        new_shape = (int(input.shape[2] / scale), int(input.shape[3] / scale))
        input = self.interpolate(input, new_shape)
        
        ########## Begin original ConvTranspose2d code ##########
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]

        x = F.conv_transpose2d(
            input, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
        
        ########## End original ConvTranspose2d code ##########

        # Match dims with desired upsampling amount
        if isinstance(self.stride, tuple):
            match_dims_shape = (int(new_shape[0] * self.stride[0]), int(new_shape[1] * self.stride[1]))
        else: # is int
            match_dims_shape = (int(new_shape[0] * self.stride), int(new_shape[1] * self.stride))

        x = self.match_dims(x, match_dims_shape)

        # Interpolate output
        out_shape = (int(x.shape[2] * scale), int(x.shape[3] * scale))

        return self.interpolate(x, out_shape)


class UNetWithScale(Unet):
    def __init__(self, base_res, scale_dict, *args, **kwargs):
        super(UNetWithScale, self).__init__(*args, **kwargs)
        self.base_res = base_res
        self.scale_dict = scale_dict
    
    def forward(self, x: torch.Tensor, **kwargs):
        self.scale_dict["scale"] = x.shape[-1] / self.base_res
        return self._forward(x)


def UNet_kernel_interpolate(model, interpolation='bilinear', device=torch.device('cuda')):
    """
    Takes existing convolutional neural network `model` replaces all Conv2d layers with Conv2d layers that
    are adaptive to the input size. Similarly replaces ConvTranspose2d with InOutInterpolateConvTranspose2d.
    """
    scale_dict = model.scale_dict
    new_model = copy.deepcopy(model)
    new_model.scale_dict = scale_dict

    def recursively_replace(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                # New conv layer with interpolated kernels
                new_conv = InterpConv2d(scale_dict, child.in_channels, child.out_channels,
                                    child.kernel_size, child.stride,
                                    child.padding, child.dilation,
                                    child.groups, bias=(child.bias is not None),
                                    interpolation_mode=interpolation)
                
                # Set the new weight and bias
                new_conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_conv.bias.data.copy_(child.bias.data)

                new_conv.to(device)
                setattr(module, name, new_conv)

            if isinstance(child, nn.ConvTranspose2d):
                new_conv = InOutInterpolateConvTranspose2d(scale_dict, child.in_channels, child.out_channels,
                                                child.kernel_size, child.stride, child.padding,
                                                child.output_padding, child.groups, bias=(child.bias is not None),
                                                dilation=child.dilation, interpolation_mode=interpolation)
                
                # Set the new weight and bias
                new_conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_conv.bias.data.copy_(child.bias.data)

                new_conv.to(device)
                setattr(module, name, new_conv)

            else:
                recursively_replace(child)

        return module
    
    
    return recursively_replace(new_model)