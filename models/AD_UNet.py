
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# # import netron
# # import torch.onnx

# # import hiddenlayer as h


# class BasicConv(nn.Module):
#     def __init__(
#         self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
#     ):
#         super(BasicConv, self).__init__()

#         self.conv = nn.Conv2d(
#             in_planes,
#             out_planes,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation,
#             bias=False,
#         )
#         self.bn = nn.BatchNorm2d(out_planes)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         return x


# class SDI(nn.Module):
#     def __init__(self, channel):
#         super().__init__()

#         self.convs = nn.ModuleList(
#             [
#                 nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
#                 for _ in range(5)
#             ]
#         )

#     def forward(self, xs, anchor):
#         ans = torch.ones_like(anchor)
#         target_size = anchor.shape[-1]

#         for i, x in enumerate(xs):
#             if x.shape[-1] > target_size:
#                 x = F.adaptive_avg_pool2d(x, (target_size, target_size))
#             elif x.shape[-1] < target_size:
#                 x = F.interpolate(
#                     x,
#                     size=(target_size, target_size),
#                     mode="bilinear",
#                     align_corners=True,
#                 )

#             ans = ans * self.convs[i](x)

#         return ans


# class EfficientChannelAttention(nn.Module):  # Efficient Channel Attention module
#     def __init__(self, c, b=1, gamma=2):
#         super(EfficientChannelAttention, self).__init__()
#         t = int(abs((math.log(c, 2) + b) / gamma))
#         k = t if t % 2 else t + 1

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.avg_pool(x)
#         x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         out = self.sigmoid(x)
#         return out


# class BasicBlock(nn.Module):

#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()

#         self.residual_function = nn.Sequential(
#             nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=3,
#                 stride=stride,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 out_channels,
#                 out_channels * BasicBlock.expansion,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion),
#         )
#         self.channel = EfficientChannelAttention(out_channels)
#         self.shortcut = nn.Sequential()

#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels,
#                     out_channels * BasicBlock.expansion,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion),
#             )

#     def forward(self, x):

#         out = self.residual_function(x)
#         eca_out = self.channel(out)
#         out = out * eca_out
#         return nn.ReLU(inplace=True)(out + self.shortcut(x))


# class ASPP(nn.Module):
#     def __init__(self, in_channel=512, depth=256):
#         super(ASPP, self).__init__()
#         # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
#         self.mean = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv = nn.Conv2d(in_channel, depth, 1, 1)
#         # k=1 s=1 no pad
#         self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
#         self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
#         self.atrous_block12 = nn.Conv2d(
#             in_channel, depth, 3, 1, padding=12, dilation=12
#         )
#         self.atrous_block18 = nn.Conv2d(
#             in_channel, depth, 3, 1, padding=18, dilation=18
#         )

#         self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

#     def forward(self, x):
#         size = x.shape[2:]

#         image_features = self.mean(x)
#         image_features = self.conv(image_features)
#         image_features = F.interpolate(image_features, size=size, mode="bilinear")

#         atrous_block1 = self.atrous_block1(x)

#         atrous_block6 = self.atrous_block6(x)

#         atrous_block12 = self.atrous_block12(x)

#         atrous_block18 = self.atrous_block18(x)

#         net = self.conv_1x1_output(
#             torch.cat(
#                 [
#                     image_features,
#                     atrous_block1,
#                     atrous_block6,
#                     atrous_block12,
#                     atrous_block18,
#                 ],
#                 dim=1,
#             )
#         )
#         return net


# class BottleNeck(nn.Module):
#     expansion = 4

#     """
#     espansion是通道扩充的比例
#     注意实际输出channel = middle_channels * BottleNeck.expansion
#     """

#     def __init__(self, in_channels, middle_channels, stride=1):
#         super().__init__()
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(middle_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 middle_channels,
#                 middle_channels,
#                 stride=stride,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(middle_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 middle_channels,
#                 middle_channels * BottleNeck.expansion,
#                 kernel_size=1,
#                 bias=False,
#             ),
#             nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
#         )

#         self.shortcut = nn.Sequential()

#         if stride != 1 or in_channels != middle_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels,
#                     middle_channels * BottleNeck.expansion,
#                     stride=stride,
#                     kernel_size=1,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
#             )

#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


# class VGGBlock(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels):
#         super().__init__()
#         self.first = nn.Sequential(
#             nn.Conv2d(in_channels, middle_channels, 3, padding=1),
#             nn.BatchNorm2d(middle_channels),
#             nn.ReLU(),
#         )
#         self.second = nn.Sequential(
#             nn.Conv2d(middle_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         out = self.first(x)
#         out = self.second(out)

#         return out


# class ADUnet(nn.Module):
#     def __init__(self, block, layers, num_classes, input_channels=2,base=32):
#         super().__init__()
#         nb_filter = [32, 64, 128, 256, 512]
#         # nb_filter = [64, 128, 256, 512,1024]
#         self.in_channel = nb_filter[0]

#         self.pool = nn.MaxPool2d(2, 2)
 

#         self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
#         self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
#         self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
#         self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
#         self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)

#         self.conv3_1 = VGGBlock(
#             (nb_filter[3] + nb_filter[4]) * block.expansion,
#             nb_filter[3],
#             nb_filter[3] * block.expansion,
#         )
#         self.conv2_2 = VGGBlock(
#             (nb_filter[2] + nb_filter[3]) * block.expansion,
#             nb_filter[2],
#             nb_filter[2] * block.expansion,
#         )
#         self.conv1_3 = VGGBlock(
#             (nb_filter[1] + nb_filter[2]) * block.expansion,
#             nb_filter[1],
#             nb_filter[1] * block.expansion,
#         )
#         self.conv0_4 = VGGBlock(
#             nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0]
#         )

#         self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

#         self.aspp1 = ASPP(32, 32)
#         self.aspp2 = ASPP(64, 64)
#         self.aspp3 = ASPP(128, 128)
#         self.aspp4 = ASPP(256, 256)
#         # self.aspp1 = ASPP(64,64)
#         # self.aspp2 = ASPP(128,128)
#         # self.aspp3 = ASPP(256,256)
#         # self.aspp4 = ASPP(512,512)
#         channel = 32
#         # channel=64
#         self.sdi_1 = SDI(channel)
#         self.sdi_2 = SDI(channel)
#         self.sdi_3 = SDI(channel)
#         self.sdi_4 = SDI(channel)
#         self.sdi_5 = SDI(channel)
#         # self.Translayer_1 = BasicConv(64, channel, 1)
#         # self.Translayer_2 = BasicConv(128, channel, 1)
#         # self.Translayer_3 = BasicConv(256, channel, 1)
#         # self.Translayer_4 = BasicConv(512, channel, 1)
#         # self.Translayer_5 = BasicConv(1024, channel, 1)

#         self.Translayer_1 = BasicConv(32, channel, 1)
#         self.Translayer_2 = BasicConv(64, channel, 1)
#         self.Translayer_3 = BasicConv(128, channel, 1)
#         self.Translayer_4 = BasicConv(256, channel, 1)
#         self.Translayer_5 = BasicConv(512, channel, 1)
#         self.deconv2 = nn.ConvTranspose2d(
#             channel, channel, kernel_size=4, stride=2, padding=1, bias=False
#         )
#         self.deconv3 = nn.ConvTranspose2d(
#             channel, channel, kernel_size=4, stride=2, padding=1, bias=False
#         )
#         self.deconv4 = nn.ConvTranspose2d(
#             channel, channel, kernel_size=4, stride=2, padding=1, bias=False
#         )
#         self.deconv5 = nn.ConvTranspose2d(
#             channel, channel, kernel_size=4, stride=2, padding=1, bias=False
#         )
#         self.seg_outs = nn.ModuleList([nn.Conv2d(channel, 1, 1, 1) for _ in range(5)])

#     def _make_layer(self, block, middle_channel, num_blocks, stride):
#         """
#         middle_channels中间维度，实际输出channels = middle_channels * block.expansion
#         num_blocks，一个Layer包含block的个数
#         """

#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channel, middle_channel, stride))
#             self.in_channel = middle_channel * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, input):
#         seg_outs = []
#         x0_0 = self.conv0_0(input)

#         x0_0 = self.aspp1(x0_0)

#         x1_0 = self.conv1_0(self.pool(x0_0))

#         x1_0 = self.aspp2(x1_0)
#         x2_0 = self.conv2_0(self.pool(x1_0))
#         x2_0 = self.aspp3(x2_0)
#         x3_0 = self.conv3_0(self.pool(x2_0))
#         x3_0 = self.aspp4(x3_0)
#         x4_0 = self.conv4_0(self.pool(x3_0))

#         x4_0 = self.Translayer_5(x4_0)
#         x3_0 = self.Translayer_4(x3_0)
#         x2_0 = self.Translayer_3(x2_0)
#         x1_0 = self.Translayer_2(x1_0)
#         x0_0 = self.Translayer_1(x0_0)

#         # f41 = self.sdi_5([x0_0, x1_0, x2_0, x3_0, x4_0], x4_0)
#         f31 = self.sdi_4([x0_0, x1_0, x2_0, x3_0, x4_0], x3_0)
#         f21 = self.sdi_3([x0_0, x1_0, x2_0, x3_0, x4_0], x2_0)
#         f11 = self.sdi_2([x0_0, x1_0, x2_0, x3_0, x4_0], x1_0)
#         f01 = self.sdi_1([x0_0, x1_0, x2_0, x3_0, x4_0], x0_0)

#         # x3_1 = self.conv3_1(torch.cat([f31, self.up(f41)], 1))
#         # x2_2 = self.conv2_2(torch.cat([f21 , self.up(x3_1)], 1))
#         # x1_3 = self.conv1_3(torch.cat([f11, self.up(x2_2)], 1))
#         # x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

#         # output = self.final(x0_4)
#         seg_outs.append(self.seg_outs[0](x4_0))

#         y = self.deconv2(x4_0) + f31
#         seg_outs.append(self.seg_outs[1](y))

#         y = self.deconv3(y) + f21
#         seg_outs.append(self.seg_outs[2](y))

#         y = self.deconv4(y) + f11
#         seg_outs.append(self.seg_outs[3](y))

#         y = self.deconv4(y) + f01
#         seg_outs.append(self.seg_outs[4](y))

#         for i, o in enumerate(seg_outs):
#             seg_outs[i] = F.interpolate(o, scale_factor=1, mode="bilinear")

#         return seg_outs[-1]

 


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import netron
# import torch.onnx

# import hiddenlayer as h


class BasicConv(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
                for _ in range(5)
            ]
        )

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(
                    x,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=True,
                )

            ans = ans * self.convs[i](x)

        return ans


class EfficientChannelAttention(nn.Module):  # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(x)
        return out


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )
        self.channel = EfficientChannelAttention(out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):

        out = self.residual_function(x)
        eca_out = self.channel(out)
        out = out * eca_out
        return nn.ReLU(inplace=True)(out + self.shortcut(x))


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=12, dilation=12
        )
        self.atrous_block18 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=18, dilation=18
        )

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode="bilinear")

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(
            torch.cat(
                [
                    image_features,
                    atrous_block1,
                    atrous_block6,
                    atrous_block12,
                    atrous_block18,
                ],
                dim=1,
            )
        )
        return net


class BottleNeck(nn.Module):
    expansion = 4

    """
    espansion是通道扩充的比例
    注意实际输出channel = middle_channels * BottleNeck.expansion
    """

    def __init__(self, in_channels, middle_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                middle_channels,
                middle_channels,
                stride=stride,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                middle_channels,
                middle_channels * BottleNeck.expansion,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != middle_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    middle_channels * BottleNeck.expansion,
                    stride=stride,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(middle_channels * BottleNeck.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(),
        )
        self.second = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.first(x)
        out = self.second(out)

        return out


class ADUnet(nn.Module):
    def __init__(self, block, layers, num_classes, input_channels=2,base=32):
        super().__init__()
        # nb_filter = [32, 64, 128, 256, 512]
        # nb_filter = [64, 128, 256, 512,1024]
        nb_filter = [base, base*2, base*4, base*8,base*16]
        self.in_channel = nb_filter[0]

        self.pool = nn.MaxPool2d(2, 2)
 

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[1], layers[0], 1)
        self.conv2_0 = self._make_layer(block, nb_filter[2], layers[1], 1)
        self.conv3_0 = self._make_layer(block, nb_filter[3], layers[2], 1)
        self.conv4_0 = self._make_layer(block, nb_filter[4], layers[3], 1)

        self.conv3_1 = VGGBlock(
            (nb_filter[3] + nb_filter[4]) * block.expansion,
            nb_filter[3],
            nb_filter[3] * block.expansion,
        )
        self.conv2_2 = VGGBlock(
            (nb_filter[2] + nb_filter[3]) * block.expansion,
            nb_filter[2],
            nb_filter[2] * block.expansion,
        )
        self.conv1_3 = VGGBlock(
            (nb_filter[1] + nb_filter[2]) * block.expansion,
            nb_filter[1],
            nb_filter[1] * block.expansion,
        )
        self.conv0_4 = VGGBlock(
            nb_filter[0] + nb_filter[1] * block.expansion, nb_filter[0], nb_filter[0]
        )

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        # self.aspp1 = ASPP(32, 32)
        # self.aspp2 = ASPP(64, 64)
        # self.aspp3 = ASPP(128, 128)
        # self.aspp4 = ASPP(256, 256)
        self.aspp1 = ASPP(base,base)
        self.aspp2 = ASPP(base*2,base*2)
        self.aspp3 = ASPP(base*4,base*4)
        self.aspp4 = ASPP(base*8,base*8)
        channel=base
        self.sdi_1 = SDI(channel)
        self.sdi_2 = SDI(channel)
        self.sdi_3 = SDI(channel)
        self.sdi_4 = SDI(channel)
        self.sdi_5 = SDI(channel)
        self.Translayer_1 = BasicConv(base, channel, 1)
        self.Translayer_2 = BasicConv(base*2, channel, 1)
        self.Translayer_3 = BasicConv(base*4, channel, 1)
        self.Translayer_4 = BasicConv(base*8, channel, 1)
        self.Translayer_5 = BasicConv(base*16, channel, 1)

      
        self.deconv2 = nn.ConvTranspose2d(
            channel, channel, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.deconv3 = nn.ConvTranspose2d(
            channel, channel, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.deconv4 = nn.ConvTranspose2d(
            channel, channel, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.deconv5 = nn.ConvTranspose2d(
            channel, channel, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.seg_outs = nn.ModuleList([nn.Conv2d(channel, 1, 1, 1) for _ in range(5)])

    def _make_layer(self, block, middle_channel, num_blocks, stride):
        """
        middle_channels中间维度，实际输出channels = middle_channels * block.expansion
        num_blocks，一个Layer包含block的个数
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, middle_channel, stride))
            self.in_channel = middle_channel * block.expansion
        return nn.Sequential(*layers)

    def forward(self, input):
        seg_outs = []
        x0_0 = self.conv0_0(input)

        x0_0 = self.aspp1(x0_0)

        x1_0 = self.conv1_0(self.pool(x0_0))

        x1_0 = self.aspp2(x1_0)
        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.aspp3(x2_0)
        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.aspp4(x3_0)
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        x4_0 = self.Translayer_5(x4_0)
        x3_0 = self.Translayer_4(x3_0)
        x2_0 = self.Translayer_3(x2_0)
        x1_0 = self.Translayer_2(x1_0)
        x0_0 = self.Translayer_1(x0_0)

        # f41 = self.sdi_5([x0_0, x1_0, x2_0, x3_0, x4_0], x4_0)
        f31 = self.sdi_4([x0_0, x1_0, x2_0, x3_0, x4_0], x3_0)
        f21 = self.sdi_3([x0_0, x1_0, x2_0, x3_0, x4_0], x2_0)
        f11 = self.sdi_2([x0_0, x1_0, x2_0, x3_0, x4_0], x1_0)
        f01 = self.sdi_1([x0_0, x1_0, x2_0, x3_0, x4_0], x0_0)

        # x3_1 = self.conv3_1(torch.cat([f31, self.up(f41)], 1))
        # x2_2 = self.conv2_2(torch.cat([f21 , self.up(x3_1)], 1))
        # x1_3 = self.conv1_3(torch.cat([f11, self.up(x2_2)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        # output = self.final(x0_4)
        seg_outs.append(self.seg_outs[0](x4_0))

        y = self.deconv2(x4_0) + f31
        seg_outs.append(self.seg_outs[1](y))

        y = self.deconv3(y) + f21
        seg_outs.append(self.seg_outs[2](y))

        y = self.deconv4(y) + f11
        seg_outs.append(self.seg_outs[3](y))

        y = self.deconv5(y) + f01
        seg_outs.append(self.seg_outs[4](y))

        for i, o in enumerate(seg_outs):
            seg_outs[i] = F.interpolate(o, scale_factor=1, mode="bilinear")
 
        return seg_outs[-1]


 
