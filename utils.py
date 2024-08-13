import torch.nn as nn
import torch.nn.functional as F
import math
kernel_initializer = 'he_uniform'


# def conv_block_2D(x, filters, block_type, repeat=1, dilation_rate=1, size=3, padding='same'):
#     result = x
#
#     for i in range(0, repeat):
#
#         if block_type == 'separated':
#             result = separated_conv2D_block(result, filters, size=size, padding=padding)
#         elif block_type == 'duckv2':
#             result = duckv2_conv2D_block(result, filters, size=size)
#         elif block_type == 'midscope':
#             result = midscope_conv2D_block(result, filters)
#         elif block_type == 'widescope':
#             result = widescope_conv2D_block(result, filters)
#         elif block_type == 'resnet':
#             result = resnet_conv2D_block(result, filters, dilation_rate)
#         elif block_type == 'conv':
#             result = Conv2D(filters, (size, size),
#                             activation='relu', kernel_initializer=kernel_initializer, padding=padding)(result)
#         elif block_type == 'double_convolution':
#             result = double_convolution_with_batch_normalization(result, filters, dilation_rate)
#
#         else:
#             return None
#
#     return result


class duckv2_conv2D_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(duckv2_conv2D_block, self).__init__()
        self.SB = separated_conv2D_block(in_channels, out_channels)
        self.MB = midscope_conv2D_block(in_channels, out_channels)
        self.WB = widescope_conv2D_block(in_channels, out_channels)
        self.RB = resnet_conv2D_block(in_channels, out_channels)
        # self.RB1 = resnet1_conv2D_block(out_channels,out_channels)
        self.BN = nn.BatchNorm2d(out_channels)


    def forward(self,x):
        x1 = self.SB(x)
        x2 = self.MB(x)
        x3 = self.WB(x)
        x4 = self.RB(x)
        # x5 = self.RB1(x4)
        # x6 = self.RB1(x5)
        x = x1 + x2 + x3 + x4
        x = self.BN(x)
        return x

# def duckv2_conv2D_block(x, filters, size):
#     x = nn.BatchNorm2d(x)
#     x1 = widescope_conv2D_block(x, filters)
#
#     x2 = midscope_conv2D_block(x, filters)
#
#     x3 = conv_block_2D(x, filters, 'resnet', repeat=1)
#
#     x4 = conv_block_2D(x, filters, 'resnet', repeat=2)
#
#     x5 = conv_block_2D(x, filters, 'resnet', repeat=3)
#
#     x6 = separated_conv2D_block(x, filters, size=6, padding='same')
#
#     x = add([x1, x2, x3, x4, x5, x6])
#
#     x = nn.BatchNorm2d(x)
#
#     return x


class separated_conv2D_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(separated_conv2D_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=1, dilation=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=1, dilation=1)
        self.GN = nn.GroupNorm(32, out_channels)
        self.SiLU = nn.SiLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.GN(x)
        x = self.SiLU(x)
        x = self.conv2(x)
        return x



class midscope_conv2D_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(midscope_conv2D_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.GN = nn.GroupNorm(32, out_channels)
        self.SiLU = nn.SiLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.GN(x)
        x = self.SiLU(x)
        x = self.conv2(x)
        return x

class widescope_conv2D_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(widescope_conv2D_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        self.GN = nn.GroupNorm(32, out_channels)
        self.SiLU = nn.SiLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.GN(x)
        x = self.SiLU(x)
        x = self.conv2(x)
        x = self.GN(x)
        x = self.SiLU(x)
        x = self.conv3(x)

        return x

class resnet_conv2D_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resnet_conv2D_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.GN = nn.GroupNorm(32, out_channels)
        self.SiLU = nn.SiLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.GN(x1)
        x1 = self.SiLU(x1)
        x2 = self.conv2(x)
        x2 = self.GN(x2)
        x2 = self.SiLU(x2)
        x2 = self.conv3(x2)
        x2 = self.GN(x2)
        x2 = self.SiLU(x2)
        x = x1 + x2
        x = self.GN(x)
        return x

class resnet1_conv2D_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resnet1_conv2D_block, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.BN = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.ReLU(x1)
        x1 = self.BN(x1)
        x2 = self.conv2(x)
        x2 = self.ReLU(x2)
        x2 = self.BN(x2)
        x2 = self.conv3(x2)
        x2 = self.ReLU(x2)
        x2 = self.BN(x2)
        x = x1 + x2
        x = self.BN(x)
        return x

class double_convolution_with_batch_normalization(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_convolution_with_batch_normalization, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.BN = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.BN(x)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.BN(x)
        return x


class PartialDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.down1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        self.up = nn.Upsample(scale_factor=2)


    def forward(self, x):
        x1_1 = self.conv1_1(x[0])
        x1 = self.down1(x1_1).mul(x[1])
        x2_1 = self.conv2_1(x1)
        x2 = self.down2(x2_1).mul(x[2])
        x3_1 = self.conv3_1(x2)
        x3 = self.down3(x3_1).mul(x[3])
        x4_1 = self.conv4_1(x3)
        x = self.up(x4_1)
        x4_2 = self.conv4_2(x)
        x = self.up(x4_2 + x3_1)
        x3_2 = self.conv3_2(x)
        x = self.up(x3_2 + x2_1)
        x2_2 = self.conv2_2(x)
        x = self.up(x2_2 + x1_1)
        x = self.conv1_2(x)
        return x




