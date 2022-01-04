import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(32, num_channels, **kwargs)


class ChannelMLP(nn.Module):
    def __init__(self, dim, norm_layer, mlp_ratio=4.):
        super().__init__()

        self.norm = norm_layer(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, mlp_dim, 1)
        self.act = Swish()
        self.fc2 = nn.Conv2d(mlp_dim, dim, 1)
        # self.apply(self.__init__weights)

    # def __init__weights(self, m):
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        shortcut = x

        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return shortcut + x


class PoolingBlock(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class TokenMixer(nn.Module):
    def __init__(self, token_mixer_type, dim, norm_layer):
        super().__init__()
        self.token_mixer_type = token_mixer_type
        if token_mixer_type == 'pooling':
            self.token_mixer = PoolingBlock()
        else:
            print('Unknown TokenMixer', self.token_mixer_type)
        self.norm = norm_layer(dim)

    def forward(self, x):
        if self.token_mixer is None or self.token_mixer_type == 'empty':
            return x

        shortcut = x

        x = self.norm(x)
        x = self.token_mixer(x)

        return shortcut + x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        stride = (stride, stride)
        padding = (padding, padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class PoolFormerBlock(nn.Module):
    def __init__(self, token_mixier_type, dim, norm_layer, pool_size=3, mlp_ratio=4):
        super().__init__()

        self.token_mixier = TokenMixer(token_mixier_type, dim, norm_layer)
        self.channel_mlp = ChannelMLP(dim, norm_layer, mlp_ratio)

    def forward(self, x):
        x = self.token_mixier(x)
        x = self.channel_mlp(x)

        return x


class PoolFormer(nn.Module):
    #  stages=[2, 2, 6, 2],
    #  stages=[2, 2, 1],
    #  stages=[2, 2],
    #  embed_dims=[64, 128, 320, 512],
    #  embed_dims=[64, 128, 256],
    #  mlp_ratios=[4, 4, 4, 4],
    #  mlp_ratios=[4, 4, 4],
    def __init__(self,
                 stages=[1, 1],
                 embed_dims=[64, 128],
                 mlp_ratios=[4, 4],
                 downsamples=[True, True],
                 width=1.0,
                 pool_size=3,
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 token_mixier_type='pooling', norm_type='batch'):
        super().__init__()

        embed_dims = [int(dim * width) for dim in embed_dims]

        if norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'layer':
            norm_layer = nn.LayerNorm
        elif norm_type == 'group':
            norm_layer = GroupNorm

        self.patch_embed = PatchEmbed(patch_size=in_patch_size,
                                      stride=in_stride,
                                      padding=in_pad,
                                      in_chans=3,
                                      embed_dim=embed_dims[0],
                                      norm_layer=norm_layer)

        network = []
        for i in range(len(stages)):
            num_layers_per_stage = stages[i]
            blocks = []
            for _ in range(num_layers_per_stage):
                blocks.append(
                    PoolFormerBlock(token_mixier_type,
                                    embed_dims[i],
                                    norm_layer,
                                    pool_size=pool_size,
                                    mlp_ratio=mlp_ratios[i])
                )
            stage = nn.Sequential(*blocks)
            network.append(stage)
            if i < len(stages) - 1 and downsamples[i]:
                network.append(
                    PatchEmbed(patch_size=down_patch_size,
                               stride=down_stride,
                               padding=down_pad,
                               in_chans=embed_dims[i],
                               embed_dim=embed_dims[i+1],
                               norm_layer=norm_layer)
                )

        self.network = nn.ModuleList(network)

        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def get_classifier(self):
        return self.head

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        for _, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        return cls_out
