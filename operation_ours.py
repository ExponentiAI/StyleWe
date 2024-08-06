import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import datetime
import numpy as np
import torch.utils.data as data

def _rescale(img):
    return img * 2.0 - 1.0

def trans_maker(size=256):
	trans = transforms.Compose([
					transforms.Resize((size+36, size+36)),
					transforms.RandomHorizontalFlip(),
					transforms.RandomCrop((size, size)),
					transforms.ToTensor(),
					_rescale
					])
	return trans


class LoadSingleDataset(Dataset):
    def __init__(self, folder_path, im_size):
        self.folder_path = folder_path
        self.item_list = sorted(os.listdir(folder_path))
        self.transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            _rescale
        ])

    def __getitem__(self, item):
        item_path = os.path.join(self.folder_path, self.item_list[item])
        return self.transform((Image.open(item_path).convert('RGB')))

    def __len__(self):
        return len(self.item_list)


class LoadMyDataset(Dataset):
    def __init__(self, rgb_folder_path, skt_folder_path, im_size):
        rgb_path_list = []
        skt_path_list = []
        rgb_name_list = sorted(os.listdir(rgb_folder_path))

        for item_name in rgb_name_list:
            rgb_path = os.path.join(rgb_folder_path, item_name)
            skt_path = os.path.join(skt_folder_path, item_name)

            rgb_path_list.append(rgb_path)
            skt_path_list.append(skt_path)

        self.skt_path_list = skt_path_list
        self.rgb_path_list = rgb_path_list

        self.data_transform = transforms.Compose([
            transforms.RandomResizedCrop((im_size, im_size), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            _rescale
        ])

    def __getitem__(self, item):
        rgb = self.data_transform(Image.open(self.rgb_path_list[item]).convert('RGB'))
        skt = self.data_transform(Image.open(self.skt_path_list[item]).convert('RGB'))
        return rgb, skt

    def __len__(self):
        return len(self.rgb_path_list)

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31


def creat_folder(save_folder, trial_name):
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    name_time = trial_name + '_' + str(nowTime)
    saved_model_folder = os.path.join(save_folder, 'fl_st_baseline_results/%s/models' % name_time)
    saved_image_folder = os.path.join(save_folder, 'fl_st_baseline_results/%s/images' % name_time)
    folders = [os.path.join(save_folder, 'fl_st_baseline_results'), os.path.join(save_folder, 'train_results/%s' % name_time),
               os.path.join(save_folder, 'fl_st_baseline_results/%s/images' % name_time),
               os.path.join(save_folder, 'fl_st_baseline_results/%s/models' % name_time)]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    return saved_model_folder, saved_image_folder


def calculate_ssim(f1, f2):
    result = 0
    for f_idx in range(len(f1)):
        result += 1 - ssim(f1[f_idx].permute(1, 2, 0).detach().cpu().numpy(),
                           f2[f_idx].permute(1, 2, 0).detach().cpu().numpy(),
                           channel_axis=-1, data_range=1.0)
    return result


def loss_for_cos(f1, f2):  # Cosine similarity
    loss_result = 0
    for f_idx in range(len(f1)):
        loss_result += 1 - F.cosine_similarity(f1[f_idx].unsqueeze(0),
                                               f2[f_idx].unsqueeze(0))
    return loss_result


def gram_matrix(input):
    # (B, C, H, W) -> (B, C, C)
    a, b, c, d = input.shape
    features = input.view(a, b, c * d)
    G = torch.bmm(features, features.transpose(2, 1))
    # normalize the values of the gram matrix
    return G.div(b * c * d)


def clip_feature(clip_model, rgb_data, skt_data, skt_gen):
    rgb_data = F.interpolate(rgb_data, size=(224, 224), mode='bilinear', align_corners=False)
    skt_data = F.interpolate(skt_data, size=(224, 224), mode='bilinear', align_corners=False)
    skt_gen = F.interpolate(skt_gen, size=(224, 224), mode='bilinear', align_corners=False)
    skt_feat = clip_model.encode_image(skt_data)
    gen_feat = clip_model.encode_image(skt_gen)

    rgb_layer = clip_forward(clip_model.visual, rgb_data.half())
    skt_layer = clip_forward(clip_model.visual, skt_gen.half())

    return skt_feat, gen_feat, rgb_layer, skt_layer


def clip_forward(vit_model, x):
    x = vit_model.conv1(x)  # shape = [*, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    x = torch.cat([vit_model.class_embedding.to(x.dtype) +
                   torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
                  dim=1)  # shape = [*, grid ** 2 + 1, width]
    x = x + vit_model.positional_embedding.to(x.dtype)
    x = vit_model.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = vit_model.transformer.resblocks[:5](x)
    x = x.permute(1, 0, 2)

    return x


def visualize_features(feature_maps):
    num_maps = feature_maps.size(1)
    num_rows = (num_maps + 7) // 8
    fig, axes = plt.subplots(num_rows, 8, figsize=(16, num_rows * 2))
    for i in range(num_maps):
        row = i // 8
        col = i % 8
        axes[row, col].imshow(feature_maps[0, i].detach().cpu().numpy(), cmap='gray')
        axes[row, col].axis('off')
    for i in range(num_maps, num_rows * 8):
        row = i // 8
        col = i % 8
        axes[row, col].axis('off')
    plt.show()


def multi_visualize(inputs):
    fig, axs = plt.subplots(len(inputs), len(inputs[0]), figsize=(16, 10))

    for i, input_data in enumerate(inputs):
        input_data_np = input_data.detach().cpu().numpy()
        for j in range(len(input_data_np)):
            axs[i, j].imshow(input_data_np[0][j])
            axs[i, j].axis('off')

    plt.show()
