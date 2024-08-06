import torch
from torch import nn
from einops import rearrange
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from torchvision.models import vgg16
import torchvision.utils as vutils
import os


#####################################################################
#                           functions                               #
#####################################################################

def Attention(content_features, style_features):
    b, c, h, w = content_features.shape
    softmax = nn.Softmax(dim=-1)

    q = content_features.view(b, c, -1)  # (b, c, h * w)
    k = style_features.view(b, c, -1)
    v = content_features.view(b, c, -1)

    att = torch.bmm(q, k.transpose(1, 2))  # (b, c, c)
    att = softmax(att)

    out = torch.bmm(att, v)  # (b, c, h * w)
    out = rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)  # (b, c, h, w)

    return out


def AdaINN(content_features, style_features, eps=1e-6):
    content_mean, content_std = torch.mean(content_features, dim=[2, 3], keepdim=True), \
                                torch.std(content_features, dim=[2, 3], keepdim=True)
    style_mean = torch.mean(style_features, dim=[0, 2, 3], keepdim=True).repeat(content_features.shape[0], 1, 1, 1)
    style_std = torch.std(style_features, dim=[0, 2, 3], keepdim=True).repeat(content_features.shape[0], 1, 1, 1)

    normalized_content_features = (content_features - content_mean) / (content_std + eps)
    normalized_features = normalized_content_features * style_std + style_mean

    return Attention(normalized_features, style_features)
    # return normalized_features


def AdaIN(content_features, style_features, eps=1e-6):
    content_mean, content_std = torch.mean(content_features, dim=[2, 3], keepdim=True), \
                                torch.std(content_features, dim=[2, 3], keepdim=True)
    style_mean = torch.mean(style_features, dim=[2, 3], keepdim=True)
    style_std = torch.std(style_features, dim=[2, 3], keepdim=True)

    normalized_content_features = (content_features - content_mean) / (content_std + eps)
    normalized_features = normalized_content_features * style_std + style_mean

    return normalized_features


def AdaIN_GAI(content_features, style_mean, style_std, eps=1e-6):
    content_mean, content_std = torch.mean(content_features, dim=[2, 3], keepdim=True), \
                                torch.std(content_features, dim=[2, 3], keepdim=True)
    style_mean = style_mean.repeat(content_features.shape[0], 1, 1, 1)
    style_std = style_std.repeat(content_features.shape[0], 1, 1, 1)

    normalized_content_features = (content_features - content_mean) / (content_std + eps)
    normalized_features = normalized_content_features * style_std + style_mean

    return normalized_features


class Adaptive_pool(nn.Module):
    '''
    take a input tensor of size: B x C' X C'
    output a maxpooled tensor of size: B x C x H x W
    '''

    def __init__(self, channel_out, hw_out):
        super().__init__()
        self.channel_out = channel_out
        self.hw_out = hw_out
        self.pool = nn.AdaptiveAvgPool2d((channel_out, hw_out ** 2))

    def forward(self, input):
        if len(input.shape) == 3:
            input.unsqueeze_(1)
        return self.pool(input).view(-1, self.channel_out, self.hw_out, self.hw_out)


#####################################################################
#                             models                                #
#####################################################################
class VGGSimple(nn.Module):
    def __init__(self):
        super(VGGSimple, self).__init__()
        self.backbone = vgg16(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, x):
        # re-normalize from [-1, 1] to [0, 1] then to the range used for vgg
        feat = (((x + 1) * 0.5) - self.norm_mean.to(x.device)) / self.norm_std.to(x.device)
        # the layer numbers used to extract features
        cut_points = [3, 8, 15, 22]
        for idx, module in enumerate(self.backbone):
            feat = module(feat)
            if idx == cut_points[0]:
                layer_1 = feat  # (B, 64, 128, 128)
            if idx == cut_points[1]:
                layer_2 = feat  # (B, 128, 64, 64)
            if idx == cut_points[2]:
                layer_3 = feat  # (B, 256, 32, 32)
            if idx == cut_points[3]:
                layer_4 = feat  # (B, 512, 16, 16)
        feat = torch.flatten(self.avgpool(feat), 1)

        return layer_1, layer_2, layer_3, layer_4, feat

# w/o MRFM
# class ConvBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, is_up=True):
#         super().__init__()
#         self.is_up = is_up
#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.block1 = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(in_channel, in_channel // 2, 3, 1, 0, bias=True)),
#             nn.BatchNorm2d(in_channel // 2),
#             nn.LeakyReLU(0.01, inplace=True)
#         )
#         self.block2 = nn.Sequential(
#             nn.ReflectionPad2d(1),
#             spectral_norm(nn.Conv2d(in_channel // 2, out_channel, 3, 1, 0, bias=True)),
#             nn.BatchNorm2d(out_channel),
#             nn.LeakyReLU(0.01, inplace=True)
#         )
#
#     def forward(self, x):
#         if self.is_up:
#             return self.block2(self.block1(self.up(x)))
#         else:
#             return self.block2(self.block1(x))
#
#
# class Generator(nn.Module):
#     def __init__(self, nfc=256, ch_out=3):
#         super(Generator, self).__init__()
#         self.decode_28 = ConvBlock(nfc * 2, nfc, is_up=True)  # 28  channel 512 -> 256
#         self.decode_56 = ConvBlock(nfc * 2, nfc // 2, is_up=True)  # 56  channel 256 -> 128
#         self.decode_112 = ConvBlock(nfc, nfc // 4, is_up=True)  # 112  channel 128 -> 64
#         self.decode_224 = ConvBlock(nfc // 2, nfc // 4, is_up=False)  # 224  channel 64 -> 32
#
#         self.final = nn.Sequential(
#             spectral_norm(nn.Conv2d(nfc // 4, ch_out, 3, 1, 1, bias=True)),
#             nn.Tanh()
#         )
#
#     def forward(self, f28, f56, f112, f224):
#         decode_28 = self.decode_28(f28)
#         decode_56 = self.decode_56(torch.cat([decode_28, f56], dim=1))
#         decode_112 = self.decode_112(torch.cat([decode_56, f112], dim=1))
#         decode_224 = self.decode_224(torch.cat([decode_112, f224], dim=1))
#
#         output = self.final(decode_224)
#         return output


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channel, in_channel, 3, 1, 0, bias=True)),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channel * 2, in_channel, 3, 1, 0, bias=True)),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=True)),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x1, x2, weight):
        f1 = self.pool(x1)
        f1 = self.block1(f1)
        out = self.block2(self.up(torch.cat([f1, x2], dim=1)))
        out = self.block3(out + weight * x1)

        return out


class Generator(nn.Module):
    def __init__(self, nfc=256, ch_out=3):
        super(Generator, self).__init__()
        self.decode_28 = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(nfc * 2, nfc, 3, 1, 0, bias=True)),
            nn.BatchNorm2d(nfc),
            nn.LeakyReLU(0.01, inplace=True)
        )  # 32  channel 512 -> 256
        self.decode_56 = ConvBlock(nfc, nfc // 2)  # 64  channel 256 -> 128
        self.decode_112 = ConvBlock(nfc // 2, nfc // 4)  # 128  channel 128 -> 64
        self.decode_224 = ConvBlock(nfc // 4, nfc // 8)  # 256  channel 64 -> 32

        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(nfc // 8, ch_out, 3, 1, 1, bias=True)),
            nn.Tanh()
        )

        self.weight_28 = nn.Parameter(torch.tensor(0.))
        self.weight_56 = nn.Parameter(torch.tensor(0.))
        self.weight_112 = nn.Parameter(torch.tensor(0.))

    def forward(self, f28, f56, f112, f224,is_train=False):
        decode_28 = self.decode_28(f28)
        decode_56 = self.decode_56(f56, decode_28, self.weight_28)
        decode_112 = self.decode_112(f112, decode_56, self.weight_56)
        decode_224 = self.decode_224(f224, decode_112, self.weight_112)

        output = self.final(decode_224)
        # print("output.size:", output.size())
        if is_train:
            return decode_28, decode_56,decode_112,decode_224,output
        return output

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride, padding):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size,stride=stride, padding=padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class ConvBlock_stu(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channel, in_channel, 3, 1, 0, bias=True)),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            DepthwiseSeparableConv2d(in_channel*2, in_channel, 3, 1, 0),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(0.01, inplace=True)
        )
        self.block3 = nn.Sequential(
            DepthwiseSeparableConv2d(in_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, x1, x2, weight):
        f1 = self.pool(x1)
        f1 = self.block1(f1)
        out = self.block2(self.up(torch.cat([f1, x2], dim=1)))
        out = self.block3(out + weight * x1)

        return out

class Generator_stu(nn.Module):
    def __init__(self, nfc=256, ch_out=3):
        super(Generator_stu, self).__init__()
        self.decode_28 = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(nfc * 2, nfc, 3, 1, 0, bias=True)),
            nn.BatchNorm2d(nfc),
            nn.LeakyReLU(0.01, inplace=True)
        )  # 32  channel 512 -> 256
        self.decode_56 = ConvBlock_stu(nfc, nfc // 2)  # 64  channel 256 -> 128
        self.decode_112 = ConvBlock_stu(nfc // 2, nfc // 4)  # 128  channel 128 -> 64
        self.decode_224 = ConvBlock_stu(nfc // 4, nfc // 8)  # 256  channel 64 -> 32

        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(nfc // 8, ch_out, 3, 1, 1, bias=True)),
            nn.Tanh()
        )

        self.weight_28 = nn.Parameter(torch.tensor(0.))
        self.weight_56 = nn.Parameter(torch.tensor(0.))
        self.weight_112 = nn.Parameter(torch.tensor(0.))

    def forward(self, f28, f56, f112, f224, is_train=False):        ## f28: 8 512 32 32     f56: 8 256 64 64     f112: 8 128 128 128     f224: 8 64 256 256
        decode_28 = self.decode_28(f28)
        decode_56 = self.decode_56(f56, decode_28, self.weight_28)
        decode_112 = self.decode_112(f112, decode_56, self.weight_56)
        decode_224 = self.decode_224(f224, decode_112, self.weight_112)

        output = self.final(decode_224)
        # print("output.size:", output.size())
        if is_train:
            return decode_28, decode_56, decode_112, decode_224, output
        return output




class DownConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, down=True):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True)),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.1),
                nn.AvgPool2d(2, 2)
            )
        else:
            self.block = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True)),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.1)
            )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, nfc=256):
        super(Discriminator, self).__init__()
        self.block1 = DownConvBlock(nfc, nfc // 2, down=False)
        self.block2 = DownConvBlock(nfc // 2, nfc // 4, down=True)
        self.out = spectral_norm(nn.Conv2d(nfc // 4, 1, 4, 2, 0))

    def forward(self, input):
        out = self.out(self.block2(self.block1(input)))
        return out.view(-1)


def train_d(net, data, label="real"):
    pred = net(data)
    if label == "real":
        loss = F.relu(1 - pred).mean()
    else:
        loss = F.relu(1 + pred).mean()

    loss.backward()


def save_images(vgg, net_g, datas, saved_image_folder, n_iter):
    net_g.eval()
    with torch.no_grad():
        feats = vgg(datas)
        imgs = net_g(feats[3], feats[2], feats[1], feats[0])
        vutils.save_image(torch.cat([datas, imgs], dim=0),
                          os.path.join(saved_image_folder, 'iter_%d.png' % (n_iter)), nrow=8)
                          # range=(-1, 1), normalize=True)
        del datas, feats, imgs
    net_g.train()
