from torch import nn

from nnm.layers.conv import ConvFFN, ConvNormAct
from nnm.layers.drop import DropPath
from nnm.layers.se import SqueezeExcitation
from nnm.layers.attention import SelfAttention
from nnm.layers.vit import LayerScale
from nnm.models.pretrained import PretrainedModel


def _groups(group_size, channels):
    if group_size == 0:
        return 1
    return channels if group_size == 1 else channels // group_size


class MobileOneBlock(nn.Module):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride=1,
        group_size=0,
        use_se=False,
        use_act=True,
        use_scale_branch=True,
        num_conv_branches=1,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.groups = _groups(group_size, in_chs)
        self.se = SqueezeExcitation(out_chs, reduction=1 / 16, activation="relu", gate="sigmoid") if use_se else nn.Identity()
        self.identity = nn.BatchNorm2d(in_chs) if out_chs == in_chs and stride == 1 else None
        self.conv_kxk = nn.ModuleList(
            [ConvNormAct(in_chs, out_chs, kernel_size, stride, groups=self.groups, act=False) for _ in range(num_conv_branches)]
        )
        self.conv_scale = (
            ConvNormAct(in_chs, out_chs, 1, stride, groups=self.groups, act=False)
            if kernel_size > 1 and use_scale_branch
            else None
        )
        self.act = act_layer() if use_act else nn.Identity()

    def forward(self, x):
        output = None
        for branch in self.conv_kxk:
            output = branch(x) if output is None else output + branch(x)
        if self.conv_scale is not None:
            output = self.conv_scale(x) if output is None else output + self.conv_scale(x)
        if self.identity is not None:
            output = self.identity(x) if output is None else output + self.identity(x)
        return self.act(self.se(output))


class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride, group_size, small_kernel=None, use_se=False, act_layer=None):
        super().__init__()
        self.groups = _groups(group_size, in_chs)
        self.large_conv = ConvNormAct(in_chs, out_chs, kernel_size, stride, groups=self.groups, act=False)
        self.small_conv = (
            ConvNormAct(in_chs, out_chs, small_kernel, stride, groups=self.groups, act=False)
            if small_kernel is not None
            else None
        )
        self.se = SqueezeExcitation(out_chs, reduction=0.25, activation="relu", gate="sigmoid") if use_se else nn.Identity()
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        output = self.large_conv(x)
        if self.small_conv is not None:
            output = output + self.small_conv(x)
        return self.act(self.se(output))


class MobileOnePatchEmbed(nn.Module):
    def __init__(self, patch_size, stride, in_chs, embed_dim, act_layer=nn.GELU, lkc_use_act=False, use_se=False):
        super().__init__()
        self.proj = nn.Sequential(
            ReparamLargeKernelConv(
                in_chs,
                embed_dim,
                patch_size,
                stride,
                group_size=1,
                small_kernel=3,
                use_se=use_se,
                act_layer=act_layer if lkc_use_act else None,
            ),
            MobileOneBlock(embed_dim, embed_dim, 1, 1, use_se=False, act_layer=act_layer),
        )

    def forward(self, x):
        return self.proj(x)


class RepConditionalPosEnc(nn.Module):
    def __init__(self, channels, spatial_shape=(7, 7)):
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = (spatial_shape, spatial_shape)
        self.pos_enc = nn.Conv2d(
            channels,
            channels,
            spatial_shape,
            1,
            int(spatial_shape[0] // 2),
            groups=channels,
            bias=True,
        )

    def forward(self, x):
        return self.pos_enc(x) + x


class RepMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm = MobileOneBlock(
            dim, dim, kernel_size, group_size=1, use_act=False, use_scale_branch=False, num_conv_branches=0
        )
        self.mixer = MobileOneBlock(dim, dim, kernel_size, group_size=1, use_act=False)
        self.layer_scale = LayerScale(dim, layer_scale_init_value, spatial=True) if layer_scale_init_value else nn.Identity()

    def forward(self, x):
        return x + self.layer_scale(self.mixer(x) - self.norm(x))


class RepMixerBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=3,
        mlp_ratio=4.0,
        drop_path=0.0,
        layer_scale_init_value=1e-5,
        proj_drop=0.0,
        head_dim=32,
    ):
        super().__init__()
        self.token_mixer = RepMixer(dim, kernel_size, layer_scale_init_value)
        self.mlp = ConvFFN(dim, int(dim * mlp_ratio), dropout=proj_drop)
        self.layer_scale = LayerScale(dim, layer_scale_init_value, spatial=True) if layer_scale_init_value else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        x = self.token_mixer(x)
        return x + self.drop_path(self.layer_scale(self.mlp(x)))


class AttentionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.0, layer_scale_init_value=1e-5, proj_drop=0.0, head_dim=32):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.token_mixer = SelfAttention(
            dim,
            dim // head_dim,
            head_dim=head_dim,
            qkv_bias=False,
            o_bias=False,
            causal=False,
        )
        self.layer_scale_1 = LayerScale(dim, layer_scale_init_value, spatial=True) if layer_scale_init_value else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path else nn.Identity()
        self.mlp = ConvFFN(dim, int(dim * mlp_ratio), dropout=proj_drop)
        self.layer_scale_2 = LayerScale(dim, layer_scale_init_value, spatial=True) if layer_scale_init_value else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        batch, channels, height, width = x.shape
        hidden = self.norm(x).flatten(2).transpose(1, 2)
        hidden = self.token_mixer(hidden).transpose(1, 2).reshape(batch, channels, height, width)
        x = x + self.drop_path1(self.layer_scale_1(hidden))
        return x + self.drop_path2(self.layer_scale_2(self.mlp(x)))


class FastViTStage(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        depth,
        token_mixer="repmixer",
        downsample=True,
        down_patch_size=7,
        down_stride=2,
        kernel_size=3,
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-5,
        proj_drop_rate=0.0,
        pos_emb=False,
        se_downsample=False,
        lkc_use_act=False,
        head_dim=32,
    ):
        super().__init__()
        self.downsample = (
            MobileOnePatchEmbed(down_patch_size, down_stride, dim, dim_out, lkc_use_act=lkc_use_act, use_se=se_downsample)
            if downsample
            else nn.Identity()
        )
        self.pos_emb = RepConditionalPosEnc(dim_out) if pos_emb else nn.Identity()
        block = AttentionBlock if token_mixer == "attention" else RepMixerBlock
        self.blocks = nn.ModuleList(
            [
                block(
                    dim=dim_out,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path_rate[index] if isinstance(drop_path_rate, (list, tuple)) else drop_path_rate,
                    layer_scale_init_value=layer_scale_init_value,
                    proj_drop=proj_drop_rate,
                    head_dim=head_dim,
                )
                for index in range(depth)
            ]
        )

    def forward(self, x):
        x = self.pos_emb(self.downsample(x))
        for block in self.blocks:
            x = block(x)
        return x


class FastViT(PretrainedModel):
    def __init__(self, config):
        super().__init__()
        in_chans = config.get("in_chans", config.get("num_channels", 3))
        layers = config.get("layers", config.get("depths", [2, 2, 6, 2]))
        embed_dims = config.get("embed_dims", config.get("dims", [64, 128, 256, 512]))
        token_mixers = config.get("token_mixers", ["repmixer"] * len(layers))
        mlp_ratios = config.get("mlp_ratios", [4.0] * len(layers))
        downsamples = config.get("downsamples", [False] + [True] * (len(layers) - 1))
        se_downsamples = config.get("se_downsamples", [False] * len(layers))
        pos_embs = config.get("pos_embs", [False] * len(layers))
        down_patch_size = config.get("down_patch_size", 7)
        down_stride = config.get("down_stride", 2)
        drop_path_rate = config.get("drop_path_rate", 0.0)
        layer_scale_init_value = config.get("layer_scale_init_value", 1e-5)
        proj_drop_rate = config.get("proj_drop_rate", 0.0)
        head_dim = config.get("head_dim", 32)
        lkc_use_act = config.get("lkc_use_act", False)
        stem_use_scale_branch = config.get("stem_use_scale_branch", True)
        cls_ratio = config.get("cls_ratio", 2.0)
        num_classes = config.get("num_classes", 0)

        self.stem = nn.Sequential(
            MobileOneBlock(in_chans, embed_dims[0], 3, 2, use_scale_branch=stem_use_scale_branch),
            MobileOneBlock(embed_dims[0], embed_dims[0], 3, 2, group_size=1, use_scale_branch=stem_use_scale_branch),
            MobileOneBlock(embed_dims[0], embed_dims[0], 1, 1, use_scale_branch=stem_use_scale_branch),
        )
        stages = []
        previous_dim = embed_dims[0]
        for index in range(len(layers)):
            rates = _stage_rates(drop_path_rate, index, layers)
            stages.append(
                FastViTStage(
                    previous_dim,
                    embed_dims[index],
                    layers[index],
                    token_mixer=token_mixers[index],
                    downsample=downsamples[index] or previous_dim != embed_dims[index],
                    down_patch_size=down_patch_size,
                    down_stride=down_stride,
                    mlp_ratio=mlp_ratios[index],
                    drop_path_rate=rates,
                    layer_scale_init_value=layer_scale_init_value,
                    proj_drop_rate=proj_drop_rate,
                    pos_emb=pos_embs[index],
                    se_downsample=se_downsamples[index],
                    lkc_use_act=lkc_use_act,
                    head_dim=head_dim,
                )
            )
            previous_dim = embed_dims[index]
        self.stages = nn.ModuleList(stages)
        self.num_features = int(embed_dims[-1] * cls_ratio)
        self.final_conv = MobileOneBlock(embed_dims[-1], self.num_features, 3, group_size=1, use_se=True)
        self.classifier = nn.Linear(self.num_features, num_classes) if num_classes else None

    def forward_features(self, pixel_values):
        hidden = self.stem(pixel_values)
        for stage in self.stages:
            hidden = stage(hidden)
        return self.final_conv(hidden)

    def forward(self, pixel_values):
        hidden = self.forward_features(pixel_values).mean((2, 3))
        return self.classifier(hidden) if self.classifier is not None else hidden


def _stage_rates(drop_path_rate, stage, layers):
    total = sum(layers)
    start = sum(layers[:stage])
    return [drop_path_rate * (start + index) / max(total - 1, 1) for index in range(layers[stage])]
