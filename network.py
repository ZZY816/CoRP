import torch
import time
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from os.path import join
from torchvision.models import vgg16_bn, resnet50
from collections import OrderedDict

np.set_printoptions(suppress=True, threshold=1e5)

"""
resize:
    Resize tensor (shape=[N, C, H, W]) to the target size (default: 224*224).
"""


def resize(input, target_size=(224, 224)):
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)


"""
weights_init:
    Weights initialization.
"""

class LatLayer(nn.Module):
    def __init__(self, in_channel, mid_channel=32):
        super(LatLayer, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.convlayer(x)
        return x


class EnLayer(nn.Module):
    def __init__(self, in_channel=32, mid_channel=32):
        super(EnLayer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.enlayer(x)
        return x





class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()

        lat_layers = []
        for idx in range(5):
            lat_layers.append(LatLayer(in_channel=in_channels[idx], mid_channel=32))
        self.lat_layers = nn.ModuleList(lat_layers)

        dec_layers = []
        for idx in range(5):
            dec_layers.append(EnLayer(in_channel=32, mid_channel=32))
        self.dec_layers = nn.ModuleList(dec_layers)

        self.top_layer = nn.Sequential(
            nn.Conv2d(in_channels[-1], 32, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

    def forward(self, feat_list ):

        feat_top = self.top_layer(feat_list[-1])

        p = feat_top
        for idx in [4, 3, 2, 1, 0]:
            p = self._upsample_add(p, self.lat_layers[idx](feat_list[idx]))
            p = self.dec_layers[idx](p)

        out = self.out_layer(p)

        return out

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear') + y


"""
Prediction:
    Compress the channel of input features to 1, then predict maps with sigmoid function.
"""


class Prediction(nn.Module):
    def __init__(self, in_channel):
        super(Prediction, self).__init__()
        self.pred = nn.Sequential(nn.Conv2d(in_channel, 1, 1), nn.Sigmoid())

    def forward(self, feats):
        pred = self.pred(feats)
        return pred


"""
Res:
    Two convolutional layers with residual structure.
"""
class Res(nn.Module):
    def __init__(self, in_channel):
        super(Res, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1),
                                  nn.BatchNorm2d(in_channel), nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channel, in_channel, 3, 1, 1))

    def forward(self, feats):
        feats = feats + self.conv(feats)
        feats = F.relu(feats, inplace=True)
        return feats


"""
Cosal_Module:
    Given features extracted from the VGG16 backbone,
    exploit SISMs to build intra cues and inter cues.
"""
class Cosal_Module(nn.Module):
    def __init__(self, H, W):
        super(Cosal_Module, self).__init__()
        self.cosal_feat = Cosal_Sub_Module(H, W)
        self.conv = nn.Sequential(nn.Conv2d(256, 128, 1), Res(128))

    def forward(self, feats, SISMs):
        # Get foreground co-saliency features.
        fore_cosal_feats = self.cosal_feat(feats, SISMs)

        # Get background co-saliency features.
        back_cosal_feats = self.cosal_feat(feats, 1.0 - SISMs)

        # Fuse foreground and background co-saliency features
        # to generate co-saliency enhanced features.
        cosal_enhanced_feats = self.conv(torch.cat([fore_cosal_feats, back_cosal_feats], dim=1))
        return cosal_enhanced_feats


"""
Cosal_Sub_Module:
  * The core module of CoRP!
"""
class Cosal_Sub_Module(nn.Module):
    def __init__(self, H, W):
        super(Cosal_Sub_Module, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(32, 128, 1), Res(128))

    def forward(self, feats, SISMs):
        N, C, H, W = feats.shape
        HW = H * W

        # Resize SISMs to the same size as the input feats.
        SISMs = resize(SISMs, [H, W])  # shape=[N, 1, H, W], SISMs are the saliency maps generated by saliency head.

        # NFs: L2-normalized features.
        NFs = F.normalize(feats, dim=1)  # shape=[N, C, H, W]

        # Co_attention_maps are utilized to filter more background noise.
        def get_co_maps(co_proxy, NFs):
            correlation_maps = F.conv2d(NFs, weight=co_proxy)  # shape=[N, N, H, W]

            # Normalize correlation maps.
            correlation_maps = F.normalize(correlation_maps.reshape(N, N, HW), dim=2)  # shape=[N, N, HW]
            co_attention_maps = torch.sum(correlation_maps , dim=1)  # shape=[N, HW]

            # Max-min normalize co-attention maps.
            min_value = torch.min(co_attention_maps, dim=1, keepdim=True)[0]
            max_value = torch.max(co_attention_maps, dim=1, keepdim=True)[0]
            co_attention_maps = (co_attention_maps - min_value) / (max_value - min_value + 1e-12)  # shape=[N, HW]
            co_attention_maps = co_attention_maps.view(N, 1, H, W)  # shape=[N, 1, H, W]
            return co_attention_maps

        # Use co-representation to obtain co-saliency features.
        def get_CoFs(NFs, co_rep):
            SCFs = F.conv2d(NFs, weight=co_rep)
            return SCFs

        # Find the co-representation proxy.
        co_proxy = F.normalize((NFs * SISMs).mean(dim=3).mean(dim=2), dim=1).view(N, C, 1, 1)  # shape=[N, C, 1, 1]

        # Reshape the co-representation proxy to compute correlations between all pixel embeddings and the proxy.
        r_co_proxy = F.normalize((NFs * SISMs).mean(dim=3).mean(dim=2).mean(dim=0), dim=0)
        r_co_proxy = r_co_proxy.view(1, C)
        all_pixels = NFs.reshape(N, C, HW).permute(0, 2, 1).reshape(N*HW, C)
        correlation_index = torch.matmul(all_pixels, r_co_proxy.permute(1, 0))

        # Employ top-K pixel embeddings with high correlation as co-representation.
        ranged_index = torch.argsort(correlation_index, dim=0, descending=True).repeat(1, C)
        co_representation = torch.gather(all_pixels, dim=0, index=ranged_index)[:32, :].view(32, C, 1, 1)

        co_attention_maps = get_co_maps(co_proxy, NFs)  # shape=[N, 1, H, W]
        CoFs = get_CoFs(NFs, co_representation)  # shape=[N, HW, H, W]
        co_saliency_feat = self.conv(CoFs * co_attention_maps)  # shape=[N, 128, H, W]

        return co_saliency_feat


"""
Refinement:
    U-net like decoder block that fuses co-saliency features and low-level features for upsampling. 
"""


class Decoder_Block(nn.Module):
    def __init__(self, in_channel):
        super(Decoder_Block, self).__init__()
        self.cmprs = nn.Conv2d(in_channel, 32, 1)
        self.merge_conv = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                                        nn.Conv2d(96, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pred = Prediction(32)

    def forward(self, low_level_feats, cosal_map, SISMs, old_feats):
        _, _, H, W = low_level_feats.shape

        cosal_map = resize(cosal_map, [H, W])
        SISMs = resize(SISMs, [H, W])
        old_feats = resize(old_feats, [H, W])

        # Predict co-saliency maps with the size of H*W.
        cmprs = self.cmprs(low_level_feats)
        new_feats = self.merge_conv(torch.cat([cmprs * cosal_map,
                                               cmprs * SISMs,
                                               old_feats], dim=1))
        new_cosal_map = self.pred(new_feats)
        return new_feats, new_cosal_map


"""
CoRP:
    The entire CoRP.
    Given a group of images and corresponding SISMs, CoRP outputs a group of co-saliency maps (predictions) at once.
"""


class CoRP(nn.Module):
    def __init__(self, backbone):
        super(CoRP, self).__init__()

        if backbone == 'vgg16':
            bb_net = list(vgg16_bn(pretrained=True).children())[0]
            bb_convs = OrderedDict({
                'conv1': bb_net[:6],
                'conv2': bb_net[6:13],
                'conv3': bb_net[13:23],
                'conv4': bb_net[23:33],
                'conv5': bb_net[33:43]
            })
            ics = [512, 512, 256, 128, 64]

        elif backbone == 'resnet50':
            bb_net = list(resnet50(pretrained=True).children())
            bb_convs = OrderedDict({
                'conv1': nn.Sequential(*bb_net[0:3]),
                'conv2': bb_net[4],
                'conv3': bb_net[5],
                'conv4': bb_net[6],
                'conv5': bb_net[7]
            })
            ics = [2048, 1024, 512, 256, 64]

        self.encoder = nn.Sequential(bb_convs)

        self.Co6 = Cosal_Module(7, 7)
        self.Co5 = Cosal_Module(14, 14)
        self.Co4 = Cosal_Module(28, 28)
        self.Co3 = Cosal_Module(56, 56)
        self.conv6_cmprs = nn.Sequential(nn.MaxPool2d(2, 2), nn.Conv2d(ics[0], 128, 1),
                                         nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                         nn.Conv2d(128, 128, 3, 1, 1))
        self.conv5_cmprs = nn.Conv2d(ics[0], 256, 1)
        self.conv4_cmprs = nn.Conv2d(ics[1], 256, 1)
        self.conv3_cmprs = nn.Conv2d(ics[2], 256, 1)

        self.merge_co_56 = Res(128)
        self.merge_co_45 = Res(128)
        self.merge_co_34 = nn.Sequential(Res(128), nn.Conv2d(128, 32, 1))
        self.get_pred_4 = Prediction(32)
        self.refine_2 = Decoder_Block(ics[3])
        self.refine_1 = Decoder_Block(ics[4])

        self.sal_decoder = Decoder(in_channels=ics[::-1])

    def forward(self, image_group, sal=None, is_training=True, gt=None):
        # Extract features from the VGG16 backbone.
        bs_group, _, _, _ = image_group.size()
        if is_training:
            bs_sal, _, _, _ = sal.size()
            group_and_sal = torch.cat((image_group, sal), dim=0)
        else:
            group_and_sal = image_group

        conv1_2 = self.encoder.conv1(group_and_sal)
        conv2_2 = self.encoder.conv2(conv1_2)
        conv3_3 = self.encoder.conv3(conv2_2)
        conv4_3 = self.encoder.conv4(conv3_3)
        conv5_3 = self.encoder.conv5(conv4_3)

        ALL_SISMs = self.sal_decoder([conv1_2, conv2_2, conv3_3, conv4_3, conv5_3])
        SISMs = ALL_SISMs[:bs_group, ...]
        if is_training:
            SISMs_sup = ALL_SISMs[bs_group:, ...]

        # Compress the channels of high-level features.
        conv6_cmprs = self.conv6_cmprs(conv5_3[:bs_group, ...])  # shape=[N, 128, 7, 7]
        conv5_cmprs = self.conv5_cmprs(conv5_3[:bs_group, ...])  # shape=[N, 256, 14, 14]
        conv4_cmprs = self.conv4_cmprs(conv4_3[:bs_group, ...])  # shape=[N, 256, 28, 28]
        conv3_cmprs = self.conv3_cmprs(conv3_3[:bs_group, ...])  # shape=[N, 128, 56, 56]
        if is_training == True:
            iterations = 1
        else:
            iterations = 3
        for i in range(iterations):
            # Obtain co-saliancy features.
            if is_training:
                maps = gt
            elif i==0:
                maps = SISMs
            cosal_feat_6 = self.Co6(conv6_cmprs, maps)  # shape=[N, 128, 7, 7]
            cosal_feat_5 = self.Co5(conv5_cmprs, maps)  # shape=[N, 128, 14, 14]
            cosal_feat_4 = self.Co4(conv4_cmprs, maps)  # shape=[N, 128, 28, 28]
            cosal_feat_3 = self.Co3(conv3_cmprs, maps)  # shape=[N, 128, 28, 28]

            # Merge co-saliancy features and predict co-saliency maps with size of 28*28 (i.e., "cosal_map_4").
            feat_56 = self.merge_co_56(cosal_feat_5 + resize(cosal_feat_6, [14, 14]))  # shape=[N, 128, 14, 14]
            feat_45 = self.merge_co_45(cosal_feat_4 + resize(feat_56, [28, 28]))  # shape=[N, 128, 28, 28]
            feat_34 = self.merge_co_34(cosal_feat_3 + resize(feat_45, [56, 56]))  # shape=[N, 128, 56, 56]
            cosal_map_4 = self.get_pred_4(feat_34)  # shape=[N, 1, 56, 56]

            # Obtain co-saliency maps with size of 224*224 (i.e., "cosal_map_1") by progressively upsampling.
            feat_23, cosal_map_2 = self.refine_2(conv2_2[:bs_group, ...], cosal_map_4, SISMs, feat_34)
            _, cosal_map_1 = self.refine_1(conv1_2[:bs_group, ...], cosal_map_4, SISMs, feat_23)
            maps = cosal_map_1

        # Return predicted co-saliency maps.
        if is_training:
            preds_list = [resize(cosal_map_4),  resize(cosal_map_2), resize(cosal_map_1)]
            preds_sal = [resize(SISMs_sup)]
            return preds_list, preds_sal
        else:
            preds = cosal_map_1
            return preds

