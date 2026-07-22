"""Global, Regional, and Local (GRL) image-restoration transformer.

Adapted from the official implementation:
https://github.com/ofsoundof/GRL-Image-Restoration
"""

from collections import OrderedDict
from collections.abc import Mapping
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.init import trunc_normal_

from neosr.archs.arch_util import net_opt, to_2tuple
from neosr.archs.grl_arch_util import (
    EfficientMixAttnTransformerBlock,
    Upsample,
    UpsampleOneStep,
    _get_stripe_info,
    bchw_to_blc,
    blc_to_bchw,
    build_last_conv,
    calculate_mask,
    calculate_mask_all,
    get_relative_coords_table_all,
    get_relative_position_index_simple,
)
from neosr.utils.registry import ARCH_REGISTRY

upscale, _ = net_opt()


class TransformerStage(nn.Module):
    """Transformer stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads_window (list[int]): Number of window attention heads in different layers.
        num_heads_stripe (list[int]): Number of stripe attention heads in different layers.
        stripe_size (list[int]): Stripe size. Default: [8, 8]
        stripe_groups (list[int]): Number of stripe groups. Default: [None, None].
        stripe_shift (bool): whether to shift the stripes. This is used as an ablation study.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qkv_proj_type (str): QKV projection type. Default: linear. Choices: linear, separable_conv.
        anchor_proj_type (str): Anchor projection type. Default: avgpool. Choices: avgpool, maxpool, conv2d, separable_conv, patchmerging.
        anchor_one_stage (bool): Whether to use one operator or multiple progressive operators to reduce feature map resolution. Default: True.
        anchor_window_down_factor (int): The downscale factor used to get the anchors.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        pretrained_window_size (list[int]): pretrained window size. This is actually not used. Default: [0, 0].
        pretrained_stripe_size (list[int]): pretrained stripe size. This is actually not used. Default: [0, 0].
        conv_type: The convolutional block before residual connection.
        init_method: initialization method of the weight parameters used to train large scale models.
            Choices: n, normal -- Swin V1 init method.
                    l, layernorm -- Swin V2 init method. Zero the weight and bias in the post layer normalization layer.
                    r, res_rescale -- EDSR rescale method. Rescale the residual blocks with a scaling factor 0.1
                    w, weight_rescale -- MSRResNet rescale method. Rescale the weight parameter in residual blocks with a scaling factor 0.1
                    t, trunc_normal_ -- nn.Linear, trunc_normal; nn.Conv2d, weight_rescale
        fairscale_checkpoint (bool): Whether to use fairscale checkpoint.
        offload_to_cpu (bool): used by fairscale_checkpoint
        args:
            out_proj_type (str): Type of the output projection in the self-attention modules. Default: linear. Choices: linear, conv2d.
            local_connection (bool): Whether to enable the local modelling module (two convs followed by Channel attention). For GRL base model, this is used.                "local_connection": local_connection,
            euclidean_dist (bool): use Euclidean distance or inner product as the similarity metric. An ablation study.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads_window,
        num_heads_stripe,
        window_size,
        stripe_size,
        stripe_groups,
        stripe_shift,
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=(0, 0),
        pretrained_stripe_size=(0, 0),
        conv_type="1conv",
        init_method="",
        fairscale_checkpoint=False,
        offload_to_cpu=False,
        args=None,
    ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.init_method = init_method

        self.blocks = nn.ModuleList()
        self.use_checkpoint = fairscale_checkpoint
        self.offload_to_cpu = offload_to_cpu
        for i in range(depth):
            block = EfficientMixAttnTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads_w=num_heads_window,
                num_heads_s=num_heads_stripe,
                window_size=window_size,
                window_shift=i % 2 == 0,
                stripe_size=stripe_size,
                stripe_groups=stripe_groups,
                stripe_type="H" if i % 2 == 0 else "W",
                stripe_shift=i % 4 in [2, 3] if stripe_shift else False,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type=qkv_proj_type,
                anchor_proj_type=anchor_proj_type,
                anchor_one_stage=anchor_one_stage,
                anchor_window_down_factor=anchor_window_down_factor,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                pretrained_stripe_size=pretrained_stripe_size,
                res_scale=0.1 if init_method == "r" else 1.0,
                args=args,
            )
            self.blocks.append(block)

        self.conv = build_last_conv(conv_type, dim)

    def _init_weights(self):
        for n, m in self.named_modules():
            if self.init_method == "w":
                if isinstance(m, (nn.Linear, nn.Conv2d)) and n.find("cpb_mlp") < 0:
                    m.weight.data *= 0.1
            elif self.init_method == "l":
                if isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 0)
            elif self.init_method.find("t") >= 0:
                scale = 0.1 ** (len(self.init_method) - 1) * int(self.init_method[-1])
                if isinstance(m, nn.Linear) and n.find("cpb_mlp") < 0:
                    trunc_normal_(m.weight, std=scale)
                elif isinstance(m, nn.Conv2d):
                    m.weight.data *= 0.1
            else:
                raise NotImplementedError(
                    f"Parameter initialization method {self.init_method} not implemented in TransformerStage."
                )

    def forward(self, x, x_size, table_index_mask):
        res = x
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                if self.offload_to_cpu:
                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        res = checkpoint.checkpoint(
                            blk, res, x_size, table_index_mask, use_reentrant=False
                        )
                else:
                    res = checkpoint.checkpoint(
                        blk, res, x_size, table_index_mask, use_reentrant=False
                    )
            else:
                res = blk(res, x_size, table_index_mask)
        res = bchw_to_blc(self.conv(blc_to_bchw(res, x_size)))

        return res + x

    def flops(self):
        pass


class GRL(nn.Module):
    r"""Image restoration transformer with global, non-local, and local connections
    Args:
        img_size (int | list[int]): Input image size. Default 64
        in_channels (int): Number of input image channels. Default: 3
        out_channels (int): Number of output image channels. Default: None
        embed_dim (int): Patch embedding dimension. Default: 96
        upscale (int): Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range (float): Image range. 1. or 255.
        upsampler (str): The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        depths (list[int]): Depth of each Swin Transformer layer.
        num_heads_window (list[int]): Number of window attention heads in different layers.
        num_heads_stripe (list[int]): Number of stripe attention heads in different layers.
        window_size (int): Window size. Default: 8.
        stripe_size (list[int]): Stripe size. Default: [8, 8]
        stripe_groups (list[int]): Number of stripe groups. Default: [None, None].
        stripe_shift (bool): whether to shift the stripes. This is used as an ablation study.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qkv_proj_type (str): QKV projection type. Default: linear. Choices: linear, separable_conv.
        anchor_proj_type (str): Anchor projection type. Default: avgpool. Choices: avgpool, maxpool, conv2d, separable_conv, patchmerging.
        anchor_one_stage (bool): Whether to use one operator or multiple progressive operators to reduce feature map resolution. Default: True.
        anchor_window_down_factor (int): The downscale factor used to get the anchors.
        out_proj_type (str): Type of the output projection in the self-attention modules. Default: linear. Choices: linear, conv2d.
        local_connection (bool): Whether to enable the local modelling module (two convs followed by Channel attention). For GRL base model, this is used.
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        pretrained_window_size (list[int]): pretrained window size. This is actually not used. Default: [0, 0].
        pretrained_stripe_size (list[int]): pretrained stripe size. This is actually not used. Default: [0, 0].
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        conv_type (str): The convolutional block before residual connection. Default: 1conv. Choices: 1conv, 3conv, 1conv1x1, linear
        init_method: initialization method of the weight parameters used to train large scale models.
            Choices: n, normal -- Swin V1 init method.
                    l, layernorm -- Swin V2 init method. Zero the weight and bias in the post layer normalization layer.
                    r, res_rescale -- EDSR rescale method. Rescale the residual blocks with a scaling factor 0.1
                    w, weight_rescale -- MSRResNet rescale method. Rescale the weight parameter in residual blocks with a scaling factor 0.1
                    t, trunc_normal_ -- nn.Linear, trunc_normal; nn.Conv2d, weight_rescale
        fairscale_checkpoint (bool): Whether to use fairscale checkpoint.
        offload_to_cpu (bool): used by fairscale_checkpoint
        euclidean_dist (bool): use Euclidean distance or inner product as the similarity metric. An ablation study.

    """

    def __init__(
        self,
        img_size=64,
        in_channels=3,
        out_channels=None,
        embed_dim=96,
        upscale=2,
        img_range=1.0,
        upsampler="",
        depths=(6, 6, 6, 6, 6, 6),
        num_heads_window=(3, 3, 3, 3, 3, 3),
        num_heads_stripe=(3, 3, 3, 3, 3, 3),
        window_size=8,
        stripe_size=(8, 8),  # used for stripe window attention
        stripe_groups=(None, None),
        stripe_shift=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="avgpool",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        out_proj_type="linear",
        local_connection=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=(0, 0),
        pretrained_stripe_size=(0, 0),
        conv_type="1conv",
        init_method="n",  # initialization method of the weight parameters used to train large scale models.
        fairscale_checkpoint=False,  # fairscale activation checkpointing
        offload_to_cpu=False,
        euclidean_dist=False,
        **kwargs,
    ):
        super().__init__()
        # Process the input arguments
        allowed_compatibility_options = {
            "double_window",
            "separable_conv_act",
            "stripe_square",
            "use_buffer",
            "use_efficient_buffer",
        }
        unknown_options = set(kwargs) - allowed_compatibility_options
        if unknown_options:
            joined = ", ".join(sorted(unknown_options))
            raise TypeError(f"Unexpected GRL options: {joined}.")
        depths = tuple(depths)
        num_heads_window = tuple(num_heads_window)
        num_heads_stripe = tuple(num_heads_stripe)
        stripe_size = tuple(stripe_size)
        stripe_groups = tuple(stripe_groups)
        pretrained_window_size = tuple(pretrained_window_size)
        pretrained_stripe_size = tuple(pretrained_stripe_size)
        if in_channels <= 0 or (out_channels is not None and out_channels <= 0):
            raise ValueError("in_channels and out_channels must be positive.")
        if embed_dim <= 0 or not depths or any(depth <= 0 for depth in depths):
            raise ValueError("embed_dim and every depth must be positive.")
        if not (len(depths) == len(num_heads_window) == len(num_heads_stripe)):
            raise ValueError(
                "depths, num_heads_window, and num_heads_stripe must have equal lengths."
            )
        if any(
            heads <= 0 or embed_dim % heads != 0
            for heads in (*num_heads_window, *num_heads_stripe)
        ):
            raise ValueError("Every attention-head count must divide embed_dim.")
        if not isinstance(window_size, int) or window_size <= 1:
            raise ValueError("window_size must be an integer greater than one.")
        if len(stripe_size) != 2 or any(
            size is not None and (not isinstance(size, int) or size <= 0)
            for size in stripe_size
        ):
            raise ValueError("stripe_size must contain two positive integers or None.")
        if len(stripe_groups) != 2 or any(
            groups is not None and (not isinstance(groups, int) or groups <= 0)
            for groups in stripe_groups
        ):
            raise ValueError(
                "stripe_groups must contain two positive integers or None."
            )
        if (
            not isinstance(anchor_window_down_factor, int)
            or anchor_window_down_factor <= 0
        ):
            raise ValueError("anchor_window_down_factor must be a positive integer.")
        if not isinstance(upscale, int) or upscale <= 0:
            raise ValueError("upscale must be a positive integer.")
        if not torch.isfinite(torch.tensor(img_range)) or img_range <= 0:
            raise ValueError("img_range must be a finite positive value.")
        if upsampler not in {"", "pixelshuffle", "pixelshuffledirect", "nearest+conv"}:
            raise ValueError(f"Unsupported upsampler: {upsampler!r}.")
        if upsampler == "nearest+conv" and upscale != 4:
            raise ValueError("nearest+conv reconstruction requires upscale=4.")
        if offload_to_cpu and not fairscale_checkpoint:
            raise ValueError("offload_to_cpu requires fairscale_checkpoint=True.")
        if not (
            0 <= drop_rate < 1 and 0 <= attn_drop_rate < 1 and 0 <= drop_path_rate < 1
        ):
            raise ValueError("Drop rates must be in the interval [0, 1).")

        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_out_feats = 64
        self.embed_dim = embed_dim
        self.upscale = upscale
        self.upsampler = upsampler
        self.img_range = img_range
        if in_channels == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        max_stripe_size = max([0 if s is None else s for s in stripe_size])
        max_stripe_groups = max([0 if s is None else s for s in stripe_groups])
        max_stripe_groups *= anchor_window_down_factor
        self.pad_size = max(window_size, max_stripe_size, max_stripe_groups)
        # if max_stripe_size >= window_size:
        #     self.pad_size *= anchor_window_down_factor
        # if stripe_groups[0] is None and stripe_groups[1] is None:
        #     self.pad_size = max(stripe_size)
        # else:
        #     self.pad_size = window_size
        self.input_resolution = to_2tuple(img_size)
        if len(self.input_resolution) != 2 or any(
            not isinstance(size, int) or size <= 0 for size in self.input_resolution
        ):
            raise ValueError("img_size must contain two positive integers.")
        self.window_size = to_2tuple(window_size)
        self.shift_size = [w // 2 for w in self.window_size]
        self.stripe_size = stripe_size
        self.stripe_groups = stripe_groups
        self.pretrained_window_size = pretrained_window_size
        self.pretrained_stripe_size = pretrained_stripe_size
        self.anchor_window_down_factor = anchor_window_down_factor

        # Head of the network. First convolution.
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)

        # Body of the network
        self.norm_start = norm_layer(embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # stochastic depth decay rule
        args = SimpleNamespace(
            out_proj_type=out_proj_type,
            local_connection=local_connection,
            euclidean_dist=euclidean_dist,
        )
        for k, v in self.set_table_index_mask(self.input_resolution).items():
            self.register_buffer(k, v, persistent=False)

        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = TransformerStage(
                dim=embed_dim,
                input_resolution=self.input_resolution,
                depth=depths[i],
                num_heads_window=num_heads_window[i],
                num_heads_stripe=num_heads_stripe[i],
                window_size=self.window_size,
                stripe_size=stripe_size,
                stripe_groups=stripe_groups,
                stripe_shift=stripe_shift,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_proj_type=qkv_proj_type,
                anchor_proj_type=anchor_proj_type,
                anchor_one_stage=anchor_one_stage,
                anchor_window_down_factor=anchor_window_down_factor,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[
                    sum(depths[:i]) : sum(depths[: i + 1])
                ],  # no impact on SR results
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                pretrained_stripe_size=pretrained_stripe_size,
                conv_type=conv_type,
                init_method=init_method,
                fairscale_checkpoint=fairscale_checkpoint,
                offload_to_cpu=offload_to_cpu,
                args=args,
            )
            self.layers.append(layer)
        self.norm_end = norm_layer(embed_dim)

        # Tail of the network
        self.conv_after_body = build_last_conv(conv_type, embed_dim)

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == "pixelshuffle":
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_out_feats, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_out_feats)
            self.conv_last = nn.Conv2d(num_out_feats, out_channels, 3, 1, 1)
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, out_channels)
        elif self.upsampler == "nearest+conv":
            # for real-world SR (less artifacts)
            assert self.upscale == 4, "only support x4 now."
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_out_feats, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.conv_up1 = nn.Conv2d(num_out_feats, num_out_feats, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_out_feats, num_out_feats, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_out_feats, num_out_feats, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_out_feats, out_channels, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1)

        self.apply(self._init_weights)
        if init_method in ["l", "w"] or init_method.find("t") >= 0:
            for layer in self.layers:
                layer._init_weights()

    def set_table_index_mask(self, x_size):
        """
        Two used cases:
        1) At initialization: set the shared buffers.
        2) During forward pass: get the new buffers if the resolution of the input changes
        """
        # ss - stripe_size, sss - stripe_shift_size
        ss, sss = _get_stripe_info(self.stripe_size, self.stripe_groups, True, x_size)
        df = self.anchor_window_down_factor

        table_w = get_relative_coords_table_all(
            self.window_size, self.pretrained_window_size
        )
        table_sh = get_relative_coords_table_all(ss, self.pretrained_stripe_size, df)
        table_sv = get_relative_coords_table_all(
            ss[::-1], self.pretrained_stripe_size, df
        )

        index_w = get_relative_position_index_simple(self.window_size)
        index_sh_a2w = get_relative_position_index_simple(ss, df, False)
        index_sh_w2a = get_relative_position_index_simple(ss, df, True)
        index_sv_a2w = get_relative_position_index_simple(ss[::-1], df, False)
        index_sv_w2a = get_relative_position_index_simple(ss[::-1], df, True)

        mask_w = calculate_mask(x_size, self.window_size, self.shift_size)
        mask_sh_a2w = calculate_mask_all(x_size, ss, sss, df, False)
        mask_sh_w2a = calculate_mask_all(x_size, ss, sss, df, True)
        mask_sv_a2w = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, False)
        mask_sv_w2a = calculate_mask_all(x_size, ss[::-1], sss[::-1], df, True)
        return {
            "table_w": table_w,
            "table_sh": table_sh,
            "table_sv": table_sv,
            "index_w": index_w,
            "index_sh_a2w": index_sh_a2w,
            "index_sh_w2a": index_sh_w2a,
            "index_sv_a2w": index_sv_a2w,
            "index_sv_w2a": index_sv_w2a,
            "mask_w": mask_w,
            "mask_sh_a2w": mask_sh_a2w,
            "mask_sh_w2a": mask_sh_w2a,
            "mask_sv_a2w": mask_sv_a2w,
            "mask_sv_w2a": mask_sv_w2a,
        }

    def get_table_index_mask(self, device=None, input_resolution=None):
        # Used during forward pass
        if input_resolution == self.input_resolution:
            return {
                "table_w": self.table_w,
                "table_sh": self.table_sh,
                "table_sv": self.table_sv,
                "index_w": self.index_w,
                "index_sh_a2w": self.index_sh_a2w,
                "index_sh_w2a": self.index_sh_w2a,
                "index_sv_a2w": self.index_sv_a2w,
                "index_sv_w2a": self.index_sv_w2a,
                "mask_w": self.mask_w,
                "mask_sh_a2w": self.mask_sh_a2w,
                "mask_sh_w2a": self.mask_sh_w2a,
                "mask_sv_a2w": self.mask_sv_a2w,
                "mask_sv_w2a": self.mask_sv_w2a,
            }
        else:
            table_index_mask = self.set_table_index_mask(input_resolution)
            for k, v in table_index_mask.items():
                table_index_mask[k] = v.to(device)
            return table_index_mask

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Only used to initialize linear layers
            # weight_shape = m.weight.shape
            # if weight_shape[0] > 256 and weight_shape[1] > 256:
            #     std = 0.004
            # else:
            #     std = 0.02
            # print(f"Standard deviation during initialization {std}.")
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.pad_size - h % self.pad_size) % self.pad_size
        mod_pad_w = (self.pad_size - w % self.pad_size) % self.pad_size
        # print("padding size", h, w, self.pad_size, mod_pad_h, mod_pad_w)

        if mod_pad_h == 0 and mod_pad_w == 0:
            return x
        mode = "reflect" if mod_pad_h < h and mod_pad_w < w else "constant"
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode)
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = bchw_to_blc(x)
        x = self.norm_start(x)
        x = self.pos_drop(x)

        table_index_mask = self.get_table_index_mask(x.device, x_size)
        for layer in self.layers:
            x = layer(x, x_size, table_index_mask)

        x = self.norm_end(x)  # B L C
        x = blc_to_bchw(x, x_size)

        return x

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"GRL expects BCHW input with {self.in_channels} channels, got {tuple(x.shape)}."
            )
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == "pixelshuffle":
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
                self.conv_up1(
                    torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
                )
            )
            x = self.lrelu(
                self.conv_up2(
                    torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
                )
            )
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            if self.in_channels == self.out_channels:
                x = x + self.conv_last(res)
            else:
                x = self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, : H * self.upscale, : W * self.upscale]

    def flops(self):
        pass

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load plain, Lightning (``model.``), or neosr-style checkpoints."""
        if isinstance(state_dict, Mapping) and isinstance(
            state_dict.get("state_dict"), Mapping
        ):
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, Mapping):
            for parameter_key in ("params_ema", "params-ema", "params"):
                if isinstance(state_dict.get(parameter_key), Mapping):
                    state_dict = state_dict[parameter_key]
                    break
        normalized = OrderedDict()
        for key, value in state_dict.items():
            if key in {"current_val_metric", "best_val_metric", "best_iter"}:
                continue
            while key.startswith("module."):
                key = key[7:]
            if key.startswith("model."):
                key = key[6:]
            if any(
                marker in key
                for marker in (
                    "relative_coords_table",
                    "relative_position_index",
                    "attn_mask",
                )
            ) or key.startswith(("table_", "index_", "mask_")):
                continue
            normalized[key] = value
        return super().load_state_dict(normalized, strict=strict, assign=assign)

    def convert_checkpoint(self, state_dict):
        state_dict = OrderedDict(state_dict)
        for k in list(state_dict):
            if (
                k.find("relative_coords_table") >= 0
                or k.find("relative_position_index") >= 0
                or k.find("attn_mask") >= 0
                or k.find("model.table_") >= 0
                or k.find("model.index_") >= 0
                or k.find("model.mask_") >= 0
                # or k.find(".upsample.") >= 0
            ):
                state_dict.pop(k)
        return state_dict


def _variant_options(variant: str, scale: int, overrides: dict) -> dict:
    common = {
        "img_size": 64,
        "upscale": scale,
        "in_channels": 3,
        "img_range": 1.0,
        "window_size": 32,
        "stripe_size": (64, 64),
        "stripe_groups": (None, None),
        "stripe_shift": True,
        "mlp_ratio": 2.0,
        "qkv_proj_type": "linear",
        "anchor_proj_type": "avgpool",
        "anchor_one_stage": True,
        "out_proj_type": "linear",
        "conv_type": "1conv",
        "init_method": "n",
    }
    if variant == "tiny":
        common.update(
            embed_dim=64,
            depths=(4, 4, 4, 4),
            num_heads_window=(2, 2, 2, 2),
            num_heads_stripe=(2, 2, 2, 2),
            upsampler="pixelshuffledirect",
            anchor_window_down_factor=4,
            local_connection=False,
        )
    elif variant == "small":
        common.update(
            embed_dim=128,
            depths=(4, 4, 4, 4),
            num_heads_window=(2, 2, 2, 2),
            num_heads_stripe=(2, 2, 2, 2),
            upsampler="pixelshuffle",
            anchor_window_down_factor=4,
            local_connection=False,
        )
    elif variant == "base":
        common.update(
            embed_dim=180,
            depths=(4, 4, 8, 8, 8, 4, 4),
            num_heads_window=(3, 3, 3, 3, 3, 3, 3),
            num_heads_stripe=(3, 3, 3, 3, 3, 3, 3),
            upsampler="pixelshuffle",
            anchor_window_down_factor=2,
            local_connection=True,
        )
    elif variant == "real":
        common.update(
            embed_dim=180,
            depths=(4, 4, 8, 8, 8, 4, 4),
            num_heads_window=(3, 3, 3, 3, 3, 3, 3),
            num_heads_stripe=(3, 3, 3, 3, 3, 3, 3),
            window_size=16,
            stripe_size=(32, 64),
            upsampler="nearest+conv",
            anchor_window_down_factor=4,
            local_connection=True,
        )
    else:
        raise ValueError(f"Unknown GRL variant: {variant}.")
    common.update(overrides)
    return common


class _GRLWrapper(nn.Module):
    def __init__(self, variant: str, selected_scale: int, kwargs: dict) -> None:
        super().__init__()
        self.model = GRL(**_variant_options(variant, selected_scale, kwargs))

    def forward(self, x):
        return self.model(x)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if isinstance(state_dict, Mapping) and isinstance(
            state_dict.get("state_dict"), Mapping
        ):
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, Mapping):
            for parameter_key in ("params_ema", "params-ema", "params"):
                if isinstance(state_dict.get(parameter_key), Mapping):
                    state_dict = state_dict[parameter_key]
                    break
        has_model_g = any(
            key.removeprefix("module.").startswith("model_g.") for key in state_dict
        )
        normalized = OrderedDict()
        for key, value in state_dict.items():
            while key.startswith("module."):
                key = key[7:]
            if has_model_g:
                if not key.startswith("model_g."):
                    continue
                key = key[8:]
            if not key.startswith("model."):
                key = f"model.{key}"
            if any(
                marker in key
                for marker in (
                    "relative_coords_table",
                    "relative_position_index",
                    "attn_mask",
                )
            ) or key.startswith(("model.table_", "model.index_", "model.mask_")):
                continue
            normalized[key] = value
        return super().load_state_dict(normalized, strict=strict, assign=assign)


@ARCH_REGISTRY.register()
class grl_tiny(_GRLWrapper):
    def __init__(
        self, upscale: int = upscale, scale: int | None = None, **kwargs
    ) -> None:
        selected_scale = upscale if scale is None else scale
        super().__init__("tiny", selected_scale, kwargs)


@ARCH_REGISTRY.register()
class grl_small(_GRLWrapper):
    def __init__(
        self, upscale: int = upscale, scale: int | None = None, **kwargs
    ) -> None:
        selected_scale = upscale if scale is None else scale
        super().__init__("small", selected_scale, kwargs)


@ARCH_REGISTRY.register()
class grl_base(_GRLWrapper):
    def __init__(
        self, upscale: int = upscale, scale: int | None = None, **kwargs
    ) -> None:
        selected_scale = upscale if scale is None else scale
        super().__init__("base", selected_scale, kwargs)


@ARCH_REGISTRY.register()
class realgrl(_GRLWrapper):
    """Official GRL-Base configuration for x4 blind/real-world SR."""

    def __init__(self, upscale: int = 4, scale: int | None = None, **kwargs) -> None:
        selected_scale = upscale if scale is None else scale
        if selected_scale != 4:
            raise ValueError(
                f"realgrl only supports the official x4 scale, got x{selected_scale}."
            )
        super().__init__("real", selected_scale, kwargs)
