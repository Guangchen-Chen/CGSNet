import torch
import torch.nn as nn
from model.semseg.deeplabv3plus import ASPPModule
from model.semseg.pspnet import PSPHead, PyramidPooling
import torch.nn.functional as F

from model.semseg.base import BaseNet


class ASPP(nn.Module):
    def __init__(self, low_level_channels, high_level_channels):
        super(ASPP, self).__init__()


        self.head = ASPPModule(high_level_channels, (12, 24, 36))

        self.reduce = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_level_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),

                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))


    def forward(self, c4, c1):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        out = torch.cat([c1, c4], dim=1)
        out = self.fuse(out)
        return out


class Deeplabv3plusEncoder(BaseNet):
    def __init__(self, backbone, nclass):
        super(Deeplabv3plusEncoder, self).__init__(backbone)
        low_level_channels = self.backbone.channels[0]
        high_level_channels = self.backbone.channels[-1]

        self.ASPP = ASPP(low_level_channels, high_level_channels)

    def base_forward(self, x):

        c1, c2, c3, c4 = self.backbone.base_forward(x)

        out = self.ASPP(c4, c1)

        return out # out=(4, 256, 80, 80)

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_module_params(self):
        return self.ASPP.parameters()


class PSPNetEncoder(BaseNet):
    def __init__(self, backbone, nclass):
        super(PSPNetEncoder, self).__init__(backbone)
        in_channels = self.backbone.channels[-1]
        inter_channels = in_channels // 4
        self.PyramidPooling = PyramidPooling(self.backbone.channels[-1])

        self.UpConv = nn.Sequential(nn.Conv2d(in_channels * 2, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU(True))

    def base_forward(self, x):

        c1, c2, c3, c4 = self.backbone.base_forward(x)
        out = self.PyramidPooling(c4)
        out = self.UpConv(out)

        return out # out=(4, 512, 40, 40)

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_module_params(self):
        return self.PyramidPooling.parameters()

    def get_up_params(self):
        return self.UpConv.parameters()




class DeepLabV2Encoder(BaseNet):
    def __init__(self, backbone, nclass):
        super(DeepLabV2Encoder, self).__init__(backbone)

        self.classifier = nn.ModuleList()
        for dilation in [6, 12, 18, 24]:
            self.classifier.append(
                nn.Conv2d(2048, nclass, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=True))

        for m in self.classifier:
            m.weight.data.normal_(0, 0.01)


    def base_forward(self, x):
        h, w = x.shape[-2:]

        x = self.backbone.base_forward(x)[-1] # (4, 2048, 40, 40)
        out = self.classifier[0](x) # (4, 6, 40, 40)
        for i in range(len(self.classifier) - 1):
            out += self.classifier[i+1](x)

        return out

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_module_params(self):
        return self.classifier.parameters()








