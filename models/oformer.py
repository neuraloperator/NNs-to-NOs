#################################### NOTE #####################################
## This file contains code based on the PR: https://github.com/neuraloperator/neuraloperator/pull/293.
## It contains our modifications of the OFormer architecture as described in the paper.
##############################  ################ ###############################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from neuralop.models.base_model import BaseModel
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.attention_kernel_integral import AttentionKernelIntegral
from neuralop.layers.channel_mlp import LinearChannelMLP


############################## UTILITY FUNCTIONS ###############################

def rotate_half(x):
    """
    Split x's channels into two equal halves.
    """
    # split the last dimension of x into two equal halves
    x = x.reshape(*x.shape[:-1], 2, -1)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    Apply rotation matrix computed based on freqs to rotate t.
    t: tensor of shape [batch_size, num_points, dim]
    freqs: tensor of shape [batch_size, num_points, 1]

    Formula: see equation (34) in https://arxiv.org/pdf/2104.09864.pdf
    """
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

def get_normalization(norm, channels):
    if norm == 'none':
        norm_fn = nn.Identity()
    elif norm == "instance_norm":
        norm_fn = nn.InstanceNorm1d(channels)
    elif norm == "group_norm":
        norm_fn = nn.GroupNorm(num_groups=32 if channels > 128 else 1, num_channels=channels)
    elif norm == 'layer_norm':
        norm_fn = nn.LayerNorm(channels)
    else:
        raise ValueError(
            f"Got norm={norm} but expected none or one of "
            "[instance_norm, group_norm, layer_norm]"
        )
    return norm_fn

def normalize(u, norm_fn):
    # transform into channel first, from: B N C to: B C N
    if isinstance(norm_fn, nn.GroupNorm) or isinstance(norm_fn, nn.InstanceNorm1d):
        u = u.permute(0, 2, 1).contiguous()
        u = norm_fn(u)
        u = u.permute(0, 2, 1).contiguous()
    else:
        u = norm_fn(u)
    return u


############################## LAYERS ###############################

class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block with n_layers of self-attention + feedforward network (FFN),
            uses pre-normalization layout.

    For the detail definition of attention-based kernel integral, see `attention_kernel_integral.py`.

    Parameters:
        in_channels : int, input channels
        out_channels : int, output channels
        hidden_channels : int, hidden channels in the attention layers and MLP layers
        num_heads : int, number of attention heads
        head_n_channels : int, dimension of each attention head
        n_layers : int, number of (attention + FFN) layers
        use_spectral_conv : bool, whether to use spectral convonlutions in parallel with attention (must be equidistant grid), default False
        spectralconv_nmodes : int, number of modes to use for spectral convolution, default 16
        use_mlp : bool, whether to use FFN after each attention layer, by default True
        mlp_dropout : float, dropout rate of the FFN, by default 0
        mlp_expansion : float, expansion factor of the FFN's hidden layer width, by default 2.0
        non_linearity : nn.Module, non-linearity module to use, by default F.gelu
        norm : string, normalization module to use, by default 'layer_norm', other available options are
            ['instance_norm', 'group_norm', 'none']
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels,
            num_heads,
            head_n_channels,
            n_layers,
            use_spectral_conv=False,
            spectralconv_nmodes=16,
            use_mlp=True,
            mlp_dropout=0,
            mlp_expansion=2.0,
            non_linearity=F.gelu,
            norm='layer_norm',
            **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_n_channels = head_n_channels
        self.n_layers = n_layers
        self.use_spectral_conv = use_spectral_conv
        self.spectralconv_modes = spectralconv_nmodes
        self.use_mlp = use_mlp
        self.mlp_dropout = mlp_dropout
        self.mlp_expansion = mlp_expansion
        self.non_linearity = non_linearity
        self.norm = norm

        self.lifting = nn.Linear(self.in_channels, self.hidden_channels) \
            if self.in_channels != self.hidden_channels else nn.Identity()

        self.to_out = nn.Linear(self.hidden_channels, self.out_channels) \
            if self.hidden_channels != self.out_channels else nn.Identity()

        self.attention_norms = nn.ModuleList([get_normalization(self.norm, self.hidden_channels) for _ in range(self.n_layers)])
        self.attention_layers = nn.ModuleList([
                                    AttentionKernelIntegral(
                                        in_channels=self.hidden_channels,
                                        out_channels=self.hidden_channels,
                                        n_heads=self.num_heads,
                                        head_n_channels=self.head_n_channels,
                                        project_query=True)
                                    for _ in range(self.n_layers)])

        if self.use_spectral_conv:
            self.conv_norms = nn.ModuleList([get_normalization(self.norm, self.hidden_channels) for _ in range(self.n_layers)])
            self.conv_layers = nn.ModuleList([
                                    SpectralConv(
                                        self.hidden_channels,
                                        self.hidden_channels,
                                        self.spectralconv_modes,
                                        rank=1.0,
                                        fixed_rank_modes=False,
                                        separable=False,
                                        factorization=None)
                                    for _ in range(self.n_layers)])

        if self.use_mlp:
            self.mlp_norms = nn.ModuleList([get_normalization(self.norm, self.hidden_channels) for _ in range(self.n_layers)])
            self.mlp_layers = nn.ModuleList([
                                    LinearChannelMLP([self.hidden_channels,
                                       int(self.hidden_channels * self.mlp_expansion),
                                       self.hidden_channels],
                                       dropout=self.mlp_dropout)
                                for _ in range(self.n_layers)])

    def forward(self,
                u,
                pos,
                pos_emb_module=None,
                grid=None,
                **kwargs):
        """
        Encode the input function u using the Transformer Encoder Block.

        Parameters:
            u: torch.Tensor, input tensor of shape [batch_size, num_grid_points, channels]
            pos: torch.Tensor, grid point coordinates of shape [batch_size, num_grid_points, channels]
            pos_emb_module: nn.Module, positional embedding module, by default None
            grid: tuple of input grid shape.
        """
        u = self.lifting(u)
        for l in range(self.n_layers):
            if self.use_spectral_conv:
                assert grid is not None
                orig_shape = u.shape

                u_attn = self.attention_layers[l](u_src=normalize(u, self.attention_norms[l]),
                                        pos_src=pos,
                                        positional_embedding_module=pos_emb_module,
                                        **kwargs)

                u_conv = normalize(u, self.conv_norms[l]).permute(0, 2, 1).reshape(-1, self.hidden_channels, *grid)
                u_conv = self.conv_layers[l](u_conv).permute(0, 2, 3, 1).reshape(*orig_shape)

                u = (u + u_conv + u_attn) / np.sqrt(3) # normalize for stability
            else:
                u_attention_skip = u
                u = self.attention_layers[l](u_src=normalize(u, self.attention_norms[l]),
                                        pos_src=pos,
                                        positional_embedding_module=pos_emb_module,
                                        **kwargs)
                u = u + u_attention_skip

            if self.use_mlp:
                u_mlp_skip = u
                u = self.mlp_layers[l](normalize(u, self.mlp_norms[l]))
                u = u + u_mlp_skip

        u = self.to_out(u)
        return u


# Note: this is not a causal-attention-based Transformer decoder as in language models
# but rather a "decoder" that maps from the latent grid to the output grid.
class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block using cross-attention to map input grid to output grid.

    For details regarding attention-based decoding, see:
    Transformer for Partial Differential Equations' Operator Learning: https://arxiv.org/abs/2205.13671
    Perceiver IO: A General Architecture for Structured Inputs & Outputs: https://arxiv.org/abs/2107.14795


    Parameters:
        n_dim: int, number of dimensions of the target domain
        in_channels : int, input channels
        out_channels : int, output channels
        hidden_channels : int, hidden channels in the attention layers and MLP layers
        num_heads : int, number of attention heads
        head_n_channels : int, dimension of each attention head
        query_basis: string, type of coordinate-based network to compute query basis function in the decoder,
            by default 'siren', other options are ['fourier', 'linear']
        use_mlp : bool, whether to use FFN after the cross-attention layer, by default True
        mlp_dropout : float, dropout rate of the FFN, by default 0
        mlp_expansion : float, expansion factor of the FFN's hidden layer width, by default 2.0
        non_linearity : nn.Module, non-linearity module to use, by default F.gelu
        norm : string, normalization module to use, by default 'layer_norm', other available options are
            ['instance_norm', 'group_norm', 'none']
        query_siren_layers: int, number of layers in SirenNet, by default 3
        query_fourier_scale: float, scale (variance) of the Gaussian Fourier Feature Transform, by default 2.0
    """

    def __init__(
            self,
            n_dim,
            in_channels,
            out_channels,
            hidden_channels,
            num_heads,
            head_n_channels,
            query_basis='siren',
            use_mlp=True,
            mlp_dropout=0,
            mlp_expansion=2.0,
            non_linearity=F.gelu,
            norm='layer_norm',
            query_siren_layers=3,
            query_fourier_scale=2.0,
            **kwargs,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_n_channels = head_n_channels
        self.use_mlp = use_mlp
        self.mlp_dropout = mlp_dropout
        self.mlp_expansion = mlp_expansion
        self.non_linearity = non_linearity
        self.norm = norm

        self.query_basis = query_basis
        self.query_siren_layers = query_siren_layers
        self.query_fourier_scale = query_fourier_scale

        self.lifting = nn.Linear(self.in_channels, self.hidden_channels) \
            if self.in_channels != self.hidden_channels else nn.Identity()

        self.out_norm = get_normalization(self.norm, self.hidden_channels)
        self.to_out = LinearChannelMLP([self.hidden_channels, self.hidden_channels, self.out_channels],
                                non_linearity=self.non_linearity)

        # build basis for decoder
        if self.query_basis == 'siren':
            self.query_basis_fn = SirenNet(dim_in=self.n_dim,
                                           dim_hidden=self.hidden_channels,
                                           dim_out=self.num_heads * self.head_n_channels,
                                           num_layers=self.query_siren_layers)
        elif self.query_basis == 'fourier':
            self.query_basis_fn = nn.Sequential(
                GaussianFourierFeatureTransform(self.n_dim,
                                                mapping_size=self.head_n_channels,
                                                scale=self.query_fourier_scale),
                nn.Linear(self.head_n_channels * 2, self.num_heads * self.head_n_channels))
        elif self.query_basis == 'linear':
            self.query_basis_fn = nn.Linear(self.n_dim, num_heads * self.head_n_channels)
        else:
            raise ValueError(f'query_basis must be one of ["siren", "fourier", "linear"], got {self.query_basis}')

        self.attention_layer = AttentionKernelIntegral(in_channels=self.hidden_channels,
                                                        out_channels=self.hidden_channels,
                                                        n_heads=self.num_heads,
                                                        head_n_channels=self.head_n_channels,
                                                        project_query=False)

    def forward(self,
                u,
                pos_src,
                pos_emb_module=None,
                pos_qry=None,
                **kwargs
                ):
        """
           Project the input function u from the source grid to the query grid using the Transformer Decoder Block.

          Parameters:
                u: torch.Tensor, input tensor of shape [batch_size, num_src_grid_points, channels]
                pos_src: torch.Tensor, grid point coordinates of shape [batch_size, num_src_grid_points, channels]
                pos_emb_module: nn.Module, positional embedding module, by default None
                pos_qry: torch.Tensor, grid point coordinates of shape [batch_size, num_sry_grid_points, channels],
                         by default None and is set to pos_src, where input and output function will be sampled on
                         the same grid (the input grid specified by pos_src).
                         If pos_qry is provided, the output function will be sampled on query grid whose coordinates
                         are specified by pos_qry. This allows the output function to be sampled on arbitrary
                         discretization.

        """
        u = self.lifting(u)
        if pos_qry is None:
            pos_qry = pos_src  # assume that the query points are the same as the source points
        query_emb = self.query_basis_fn(pos_qry)
        query_emb = query_emb.view(pos_qry.shape[0], -1, self.num_heads * self.head_n_channels)
        if query_emb.shape[0] != u.shape[0]:
            query_emb = query_emb.expand(u.shape[0], -1, -1)

        u_out = self.attention_layer(u_src=u,
                                     pos_src=pos_src,
                                     u_qry=query_emb,
                                     pos_qry=pos_qry,
                                     positional_embedding_module=pos_emb_module,
                                     **kwargs)
        u_out = self.to_out(normalize(u_out, self.out_norm))
        return u_out

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        """
        Applying rotary positional embedding (https://arxiv.org/abs/2104.09864) to the input feature tensor.
        Code modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py

        The crux is the dot product of two rotation matrices R(theta1) and R(theta2) is equal to R(theta2 - theta1).
        """
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, coordinates):
        """coordinates is tensor of [batch_size, num_points]"""
        coordinates = coordinates * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', coordinates, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]

    @staticmethod
    def apply_1d_rotary_pos_emb(t, freqs):
        return apply_rotary_pos_emb(t, freqs)

    @staticmethod
    def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
        """Split the last dimension of features into two equal halves
           and apply 1d rotary positional embedding to each half."""
        d = t.shape[-1]
        t_x, t_y = t[..., :d//2], t[..., d//2:]

        return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                          apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)


class GaussianFourierFeatureTransform(nn.Module):
    def __init__(self,
                 in_channels,
                 mapping_size,
                 scale=10,
                 learnable=False):
        """
        An implementation of Gaussian Fourier feature mapping,
            code modified from: https://github.com/ndahlquist/pytorch-fourier-feature-networks
        "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
            https://arxiv.org/abs/2006.10739
            https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
        Given an input of size [batches, n_points, num_input_channels],
           returns a tensor of size [batches, n_points, mapping_size*2].

        Parameters:
            in_channels: int, Number of input channels.
            mapping_size: int, Number of output channels for sin/cos part.
            scale: float, Scale (variance) of the Gaussian Fourier Feature Transform, by default 10.
            learnable: bool, Whether the Gaussian projection matrix is learnable or not, by default False.

        """
        super().__init__()

        self._B = nn.Parameter(torch.randn((in_channels, mapping_size)) * scale,
                               requires_grad=learnable)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = x.view(-1, channels)

        x = x @ self._B.to(x.device)

        # From [(B*N), C] to [B, N, C]
        x = x.view(batches, num_of_points, -1)

        x = 2 * torch.pi * x

        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    
class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class Siren(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 w0=1.,
                 c=6.,
                 is_first=False,
                 use_bias=True,
                 activation=None):
        """
            code modified from:
                 https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
            SIREN paper: https://arxiv.org/abs/2006.09661
            The Siren layer is a linear layer followed by a sine activation function.

            Parameters:
                dim_in: int, Number of input channels.
                dim_out: int, Number of output channels.
                w0: float, scaling factor (denominator) used to initialize the weights, by default 6.
                c: float, scaling factor (numerator) used to initialize the weights, by default 6.
                is_first: bool, Whether this is the first layer of the network, by default False.
                use_bias: bool, Whether to use bias or not, by default True.
                activation: nn.Module, Activation function to use, by default None (which uses Sine activation).

        """

        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class SirenNet(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_hidden,
                 dim_out,
                 num_layers,
                 w0=1.,
                 w0_initial=30.,
                 use_bias=True,
                 final_activation=None):
        """
            A MLP network with Siren layers.

            Parameters:
                dim_in: int, Number of input channels.
                dim_hidden: int, Number of hidden channels.
                dim_out: int, Number of output channels.
                num_layers: int, Number of layers in the network.
                w0: float, scaling factor (denominator) used to initialize the weights, by default 6.
                w0_initial: float, scaling factor (denominator) used to initialize the weights for the first layer, by default 30.
                use_bias: bool, Whether to use bias or not, by default True.
                final_activation: nn.Module, Activation function to use in the final layer, by default None.
        """
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first,
            ))

        self.final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = Siren(dim_in=dim_hidden,
                                dim_out=dim_out,
                                w0=w0,
                                use_bias=use_bias,
                                activation=final_activation)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        x = self.last_layer(x)
        x = self.final_activation(x)
        return x


############################## MODELS ###############################

class OFormer(BaseModel, name='oformer'):
    """N-Dimensional Transformer-based Neural Operator (currently does not support N>2)
        using softmax-free attention to compute the kernel integral.

        Each layer in the encoder part is organized as follow (a.k.a pre-norm version of Transformer layer):
            u = attn(norm(u)) + u
            u = mlp(norm(u)) + u
        where u is the input function to the layer.

        For the decoder (cross-attention), given query bases q and src function u:
            u_out = attn(q, u)   # u_out will has the same shape as q


        This architecture has the modification that the encoder and decoder may include
        SpectralConv branches in parallel with every attention layer.

        Parameters
        ----------
        n_dim : int
            Number of dimensions of the domain
        in_channels : int, optional
            Number of input channels, by default 1
        out_channels : int, optional
            Number of output channels, by default 1
        encoder_hidden_channels : int
            Width of the encoder (i.e. number of channels in attention layer and MLP)
        use_decoder : bool
            Whether to use decoder. Only to be used if the input and output grids are identical.
        decoder_hidden_channels : int
            Width of the decoder (i.e. number of channels in attention layer and MLP)
        encoder_num_heads: int, optional
            Number of heads in the encoder attention, by default 1
        decoder_num_heads: int, optional
            Number of heads in the decoder cross-attention, by default 8
        encoder_head_n_channels: int, optional
            Dimension of each attention head in the encoder, by default equals to encoder_hidden_channels
        decoder_head_n_channels: int, optional
            Dimension of each attention head in the decoder, by default equals to decoder_hidden_channels
        encoder_n_layers : int, optional
            Number of Transformer layer in the encoder, by default 3
        spectral_conv_encoder : bool, optional
            Whether to include spectral convolutional layers in the encoder, by default False
        num_modes : int, optional
            The number of modes to use in the spectral convolution (if true above), by default 16
        query_basis: string, optional
            Type of coordinate-based network to compute query basis function in the decoder,
            by default 'siren', other options are ['fourier', 'linear']
        query_siren_layers: int, optional
            Number of layers in SirenNet, by default 4
        query_fourier_scale: float, optional
            Scale of the Gaussian Fourier Feature Transform in random Fourier Feature, by default 2.0
        use_mlp : bool, optional
            Whether to use an MLP layer after each attention block, by default True
        mlp_dropout : float , optional
            droupout parameter of MLP layer, by default 0
        mlp_expansion : float, optional
            expansion parameter of MLP layer, by default 2.0
        non_linearity : nn.Module, optional
            Non-Linearity module to use, by default F.gelu
        norm: string, optional
            Normalization module to modeluse, by default layernorm

        """
    def __init__(self,
                 n_dim,
                 in_channels=1,
                 out_channels=1,
                 encoder_hidden_channels=128,
                 use_decoder=True,
                 decoder_hidden_channels=128,
                 encoder_num_heads=1,
                 decoder_num_heads=8,
                 encoder_head_n_channels=None,
                 decoder_head_n_channels=None,
                 encoder_n_layers=3,
                 spectral_conv_encoder=False,
                 num_modes=16,
                 query_basis='siren',
                 query_siren_layers=4,          # number of layers in SirenNet
                 query_fourier_scale=2.0,       # scale of the Gaussian Fourier Feature Transform
                 pos_emb='rotary',              # ['rotary', 'none']
                 use_mlp=True,
                 mlp_dropout=0,
                 mlp_expansion=2.0,
                 non_linearity=F.gelu,
                 norm='layer_norm',      # ['layer_norm', 'instance_norm', ''group_norm', 'none']
                ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_dim = n_dim
        self.encoder_num_heads = encoder_num_heads
        self.decoder_num_heads = decoder_num_heads
        self.use_decoder = use_decoder
        self.encoder_hidden_channels = encoder_hidden_channels
        self.decoder_hidden_channels = decoder_hidden_channels
        self.encoder_head_n_channels = encoder_head_n_channels if encoder_head_n_channels is not None else encoder_hidden_channels
        self.decoder_head_n_channels = decoder_head_n_channels if decoder_head_n_channels is not None else decoder_hidden_channels
        self.encoder_n_layers = encoder_n_layers
        self.query_basis = query_basis
        self.query_siren_layers = query_siren_layers
        self.query_fourier_scale = query_fourier_scale
        self.pos_emb = pos_emb
        
        self.spectral_conv_encoder = spectral_conv_encoder
        self.n_modes = (num_modes,) * n_dim

        self.use_mlp = use_mlp
        self.mlp_dropout = mlp_dropout
        self.mlp_expansion = mlp_expansion
        self.non_linearity = non_linearity
        self.norm = norm

        if self.pos_emb not in ['rotary', 'none']:
            raise ValueError(f'pos_emb must be one of ["rotary", "none"], got {self.pos_emb}')

        if self.pos_emb == 'rotary':
            self.enc_pos_emb_module = RotaryEmbedding(self.encoder_head_n_channels // self.n_dim)
            self.dec_pos_emb_module = RotaryEmbedding(self.decoder_head_n_channels // self.n_dim)
        else:
            self.enc_pos_emb_module = None
            self.dec_pos_emb_module = None

        self.encoder = TransformerEncoderBlock(
                            in_channels=self.in_channels,
                            out_channels=self.decoder_hidden_channels if self.use_decoder else self.out_channels,
                            hidden_channels=self.encoder_hidden_channels,
                            num_heads=self.encoder_num_heads,
                            head_n_channels=self.encoder_head_n_channels,
                            n_layers=self.encoder_n_layers,
                            query_basis=self.query_basis,
                            use_spectral_conv=self.spectral_conv_encoder,
                            spectralconv_nmodes=self.n_modes,
                            use_mlp=self.use_mlp,
                            mlp_dropout=self.mlp_dropout,
                            mlp_expansion=self.mlp_expansion,
                            non_linearity=self.non_linearity,
                            query_siren_layers=self.query_siren_layers,
                            query_fourier_scale=self.query_fourier_scale,
                            norm=self.norm,
                        )
        
        if self.use_decoder:
            self.decoder = TransformerDecoderBlock(
                n_dim=self.n_dim,
                in_channels=self.decoder_hidden_channels,
                out_channels=self.out_channels,
                hidden_channels=self.decoder_hidden_channels,
                num_heads=self.decoder_num_heads,
                head_n_channels=self.decoder_head_n_channels,
                query_basis=self.query_basis,
                use_mlp=self.use_mlp,
                mlp_dropout=self.mlp_dropout,
                mlp_expansion=self.mlp_expansion,
                non_linearity=self.non_linearity,
                query_siren_layers=self.query_siren_layers,
                query_fourier_scale=self.query_fourier_scale,
                norm=self.norm,
            )

    def forward(self,
                u,
                pos_src,
                pos_qry=None,
                grid=None,
                **kwargs):
        """Transformer NO's forward pass,
           please note that coordinates must be normalized to [-1, 1] interval when using siren"""

        # encoder part, use self-attention to process input function
        u = self.encoder(u, pos_src, self.enc_pos_emb_module, grid, **kwargs)

        # decoder part, use cross-attention to query the solution function
        if self.use_decoder:
            u = self.decoder(u, pos_src, self.dec_pos_emb_module, pos_qry, **kwargs)

        return u

### The following wrapper is to maintain a consistent input format with other FNO-based model
class OFormerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, **samples):
        grid = x.shape[-2:]
        x = x.permute(0, 2, 3, 1)  # channel first to channel last
        nx, ny = x.shape[2], x.shape[1]
        input_pos_x, input_pos_y = torch.meshgrid(
            [torch.linspace(0, 1, x.shape[1]),
            torch.linspace(0, 1, x.shape[2])])
        x = x.reshape(x.shape[0], -1, 1)
        input_pos = torch.stack([input_pos_x, input_pos_y], dim=-1).reshape(1, -1, 2).to(x.device)
        input_pos = input_pos.repeat(x.shape[0], 1, 1)
        y_pred = self.model(x, input_pos, grid=grid)
        y_pred = y_pred.reshape(y_pred.shape[0], ny, nx, -1).permute(0, 3, 1, 2)
        return y_pred
