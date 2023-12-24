'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary
import torch
from model.dht import DHT_Layer


class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


# class MobileCon(nn.Module)
#     def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride=1):
#         super(MobileCon,self).__init__()
#         self.stride = stride
#
#         self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(expand_size)
#         self.act1 = act(inplace=True)
#
#         self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
#                                padding=kernel_size // 2, groups=expand_size, bias=False)
#         self.bn2 = nn.BatchNorm2d(expand_size)
#         self.act2 = act(inplace=True)
#
#         self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_size)
#         self.act3 = act(inplace=True)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


class MobileNetV3_Large(nn.Module):
    def __init__(self, numAngle, numRho, num_classes=1000, act=nn.Hardswish):
        super(MobileNetV3_Large, self).__init__()
        self.numAngle = numAngle
        self.numRho = numRho
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.bneck1 = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, False, 1),
            Block(3, 16, 64, 24, nn.ReLU, False, 2),
            Block(3, 24, 72, 24, nn.ReLU, False, 1),

        )
        self.bneck2 = nn.Sequential(
            Block(5, 24, 72, 40, nn.ReLU, True, 2),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
        )
        self.bneck3 = nn.Sequential(
            Block(3, 40, 240, 80, act, False, 2),
            Block(3, 80, 200, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 480, 112, act, True, 1),
            Block(3, 112, 672, 112, act, True, 1),
        )
        self.bneck4 = nn.Sequential(
            Block(5, 112, 672, 160, act, True, 2),
            Block(5, 160, 672, 160, act, True, 1),
            Block(5, 160, 960, 160, act, True, 1),
        )

        self.toplayer = nn.Conv2d(112, 24, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(40, 24, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(16, 24, kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self.smooth = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)

        self.dht_detector1 = DHT_Layer(24, 32, numAngle=self.numAngle, numRho=self.numRho)
        self.dht_detector2 = DHT_Layer(24, 32, numAngle=self.numAngle, numRho=self.numRho // 2)
        self.dht_detector3 = DHT_Layer(24, 32, numAngle=self.numAngle, numRho=self.numRho // 4)
        self.dht_detector4 = DHT_Layer(24, 32, numAngle=self.numAngle, numRho=self.numRho // 4)

        self.last_conv = nn.Sequential(
            nn.Conv2d(512, 1, 1)
        )
        self.con_p1 = nn.Conv2d(32, 1, 1)
        self.con_p2 = nn.Conv2d(32, 1, 1)
        self.con_p3 = nn.Conv2d(32, 1, 1)
        self.con_p4 = nn.Conv2d(32, 1, 1)


        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(960, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)

        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def sample_cat(self, p1, p2, p3, p4):
        p1 = nn.functional.interpolate(p1, size=(self.numAngle, self.numRho), mode='bilinear')
        p2 = nn.functional.interpolate(p2, size=(self.numAngle, self.numRho), mode='bilinear')
        p3 = nn.functional.interpolate(p3, size=(self.numAngle, self.numRho), mode='bilinear')
        p4 = nn.functional.interpolate(p4, size=(self.numAngle, self.numRho), mode='bilinear')
        # return torch.cat([p1, p2, p3, p4], dim=1)
        return p1, p2, p3, p4

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out1 = self.bneck1(out)
        out2 = self.bneck2(out1)
        out3 = self.bneck3(out2)
        # out4 = self.bneck4(out3)

        out3 = self.toplayer(out3)
        out2 = nn.functional.upsample(out3, size=out2.size()[2:], mode='bilinear') + self.latlayer1(out2)
        out1 = nn.functional.upsample(out2, size=out1.size()[2:], mode='bilinear') + self.latlayer2(out1)
        out = nn.functional.upsample(out1, size=out.size()[2:], mode='bilinear') + self.latlayer3(out)

        out = self.smooth(out)
        out1 = self.smooth1(out1)
        out2 = self.smooth2(out2)

        p1 = self.dht_detector1(out)
        p2 = self.dht_detector2(out1)
        p3 = self.dht_detector3(out2)
        p4 = self.dht_detector4(out3)

        p1, p2, p3, p4 = self.sample_cat(p1, p2, p3, p4)


        p1 = self.con_p1(p1)
        p2 = self.con_p1(p2)
        p3 = self.con_p1(p3)
        p4 = self.con_p1(p4)

        return p1, p2, p3, p4

#
# model = MobileNetV3_Large(100,100)
# # print(model)
# summary(model,(3,224,224))
