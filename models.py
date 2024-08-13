from functools import partial
import numpy as np

import torch
from torch import nn

from timm.models.vision_transformer import _cfg
from .dcn import DeformConv2d
from .utils import separated_conv2D_block, midscope_conv2D_block, widescope_conv2D_block, resnet_conv2D_block, PartialDecoder


class Mlp(nn.Module):
    def __init__(self, in_channels, hidden_dim = 2048):
        super().__init__()
        # self.fc1 = nn.Conv2d(H, hidden_dim, kernel_size=3, padding=1)
        # self.fc2 = nn.Conv2d(hidden_dim, H, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_channels)
        self.act_fn = nn.GELU()
        self.GN = nn.GroupNorm(32, in_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x1 = x.permute(0,3,2,1)
        x1 = self.fc1(x1)
        x1 = self.GN(x1)
        x1 = self.act_fn(x1)
        x1 = self.fc2(x1)
        x = x1.permute(0,3,2,1)
        return x


class Transformer2Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.DCN = DeformConv2d(in_channels, out_channels)
        self.BN = nn.BatchNorm2d(out_channels)
        self.EF = EnhancedFeatureExtraction(out_channels, out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)
        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.BN(self.DCN(x))
        x1 = self.skip(x) + x1
        x2 = self.BN(self.EF(x1))
        x = x1 + x2
        x = self.down(x)
        return x

class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            in_layer(in_channels, out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            out_layer(out_channels, out_channels),
        )
        # self.down = nn.Conv2d(out_channels ,out_channels, kernel_size=3, padding=1, stride=2)
        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        x1 = h + self.skip(x)
        h = self.out_layers(x1)
        # x = self.down(h + x1)
        return h + x1

class in_layer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        super().__init__()
        self.spe = separated_conv2D_block(in_channels, out_channels)
        self.res = resnet_conv2D_block(in_channels, out_channels)
        self.res1 = resnet_conv2D_block(out_channels, out_channels)

    def forward(self, x):
        x1 = self.spe(x)
        x2 = self.res(x)
        x = x1 + x2
        return x

class EnhancedFeatureExtraction(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        super().__init__()
        self.wid = widescope_conv2D_block(in_channels, out_channels)
        self.mid = midscope_conv2D_block(in_channels, out_channels)
        self.spe = separated_conv2D_block(in_channels, out_channels)



    def forward(self, x):
        x0 = self.wid(x)
        x1 = self.mid(x)
        x2 = self.spe(x)
        return x0 + x1 + x2

class out_layer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
    ):
        super().__init__()
        self.wid = widescope_conv2D_block(in_channels, out_channels)
        self.mid = midscope_conv2D_block(in_channels, out_channels)

    def forward(self, x):
        x0 = self.wid(x)
        x1 = self.mid(x)
        return x0 + x1

class UMF(nn.Module):
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 2, 4, 8],
        n_levels_down=4,
        n_levels_up=4,
        n_RBs=1,
        in_resolution=352,
    ):

        super().__init__()

        self.enc_blocks = nn.ModuleList()
        self.global_branch = nn.ModuleList(
            [Transformer2Conv(in_channels, min_level_channels)]
        )
        self.res_branch = nn.ModuleList(
            [nn.Conv2d(in_channels, min_level_channels, kernel_size=3, stride=2, padding=1)]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]

            for block in range(n_RBs):
                if level == 0:
                    self.enc_blocks.append(
                        nn.Sequential(nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1), RB(ch, min_channel_mult * min_level_channels), RB(min_channel_mult * min_level_channels, min_channel_mult * min_level_channels), nn.Conv2d(min_channel_mult * min_level_channels, min_channel_mult * min_level_channels, kernel_size=3, padding=1, stride=2))
                    )
                else:
                    self.enc_blocks.append(
                        nn.Sequential(RB(ch*2, min_channel_mult * min_level_channels), RB(min_channel_mult * min_level_channels, min_channel_mult * min_level_channels), nn.Conv2d(min_channel_mult * min_level_channels, min_channel_mult * min_level_channels, kernel_size=3, padding=1, stride=2))
                    )
                if level > 0:
                    self.global_branch.append(
                        nn.Sequential(Transformer2Conv(ch, min_channel_mult * min_level_channels))
                    )
                    self.res_branch.append(
                        nn.Sequential(
                            nn.Conv2d(ch, min_channel_mult * min_level_channels, kernel_size=3, stride=2, padding=1))
                    )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(RB(ch*2, ch), RB(ch, ch))

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]
            if level<3:
                ch = min_channel_mult * min_level_channels
                layers = [RB(ch + enc_block_chans.pop(),min_channel_mult * min_level_channels),RB(min_channel_mult * min_level_channels,(min_channel_mult * min_level_channels)//2)]
                layers.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),nn.Conv2d(ch//2, ch//2, kernel_size=3, padding=1),))
            else:
                layers = [RB(64, 32), nn.Upsample(scale_factor=2, mode="nearest"),nn.Conv2d(ch//2, ch//2, kernel_size=3, padding=1)]
            self.dec_blocks.append(nn.Sequential(*layers))

        self.FFN = PartialDecoder()

    def forward(self, x):
        res = []
        res1 = []
        hs = []
        h = x
        r = x
        r1 = x
        for module in self.global_branch:
            r = module(r)
            res.append(r)
        for module in self.res_branch:
            r1 = module(r1)
            res1.append(r1)
        Out_glo = self.FFN(res)
        Out_res = self.FFN(res1)
        res1 = res1[::-1]
        res = res[::-1]
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
            # h = h + res1.pop()
            # h = torch.cat([h, res.pop()], dim=1)
            h = torch.cat([h+res1.pop(), res.pop()], dim=1)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        return h, Out_glo, Out_res

class UMFNet(nn.Module):
    def __init__(self, size=352):

        super().__init__()

        self.UMF = UMF(in_resolution=size)
        self.PH = nn.Sequential(
            RB(32, 64), RB(64, 64), nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x2, glo_1, res_1 = self.UMF(x)

        out = self.PH(x2)

        return out, glo_1, res_1

