from torch.utils.data import DataLoader
import clip
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np


USE_FEDBN: bool = True
isEpoch_tr: bool = True
isEpoch_te: bool = True


import os
import time
import lpips
import torch, gc
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as Dataset
import torchvision.utils as vutils
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from model_ours import Generator, VGGSimple, Discriminator, train_d, Adaptive_pool, AdaIN_GAI
from operation_ours import LoadMyDataset, LoadSingleDataset, creat_folder, loss_for_cos, gram_matrix, calculate_ssim, \
    clip_feature, visualize_features, multi_visualize, trans_maker, InfiniteSamplerWrapper
import numpy
import argparse
import tqdm
import wandb
from metrics import calculate_fid_modify
import datetime
import random
from FedDecorr import FedDecorrLoss

seed = 24
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_data():
    trainset_A = Dataset.ImageFolder(root=data_root_A, transform=trans_maker(args.im_size))
    trainset_B = Dataset.ImageFolder(root=data_root_B, transform=trans_maker(args.im_size))
    trainset = ConcatDataset([trainset_A, trainset_B])

    trainloader = iter(DataLoader(trainset, args.batch_size, shuffle=False, \
                                   sampler=InfiniteSamplerWrapper(dataset_A), num_workers=0, pin_memory=False))
    testloader =  DataLoader(trainset_A, 12, shuffle=False, num_workers=0)
    num_examples = {"trainset" : len(trainset), "testset" : len(trainset_A)}
    return trainloader, testloader, num_examples

def save_image(net_g, dataloader, saved_image_folder, n_iter):
    net_g.eval()
    with torch.no_grad():
        imgs = []
        real = []
        for i, d in enumerate(dataloader):
            if i < 2:

                real_style = next(dataloader_B)[0].to(device)
                sf_1, sf_2, sf_3, sf_4, feat_skt = vgg(real_style)
                mean_1, std_1 = torch.mean(sf_1, dim=[0, 2, 3], keepdim=True), \
                                torch.std(sf_1, dim=[0, 2, 3], keepdim=True)
                mean_2, std_2 = torch.mean(sf_2, dim=[0, 2, 3], keepdim=True), \
                                torch.std(sf_2, dim=[0, 2, 3], keepdim=True)
                mean_3, std_3 = torch.mean(sf_3, dim=[0, 2, 3], keepdim=True), \
                                torch.std(sf_3, dim=[0, 2, 3], keepdim=True)
                mean_4, std_4 = torch.mean(sf_4, dim=[0, 2, 3], keepdim=True), \
                                torch.std(sf_4, dim=[0, 2, 3], keepdim=True)
                rf_1, rf_2, rf_3, rf_4, feat_rgb = vgg(d[0].to(device))
                rf_4, rf_3, rf_2, rf_1 = AdaIN_GAI(rf_4, mean_4, std_4), AdaIN_GAI(rf_3, mean_3, std_3), \
                                         AdaIN_GAI(rf_2, mean_2, std_2), AdaIN_GAI(rf_1, mean_1, std_1)

                imgs.append(net_g(rf_4, rf_3, rf_2, rf_1).cpu())
                real.append(d[0])
                gc.collect()
                torch.cuda.empty_cache()
            else:
                break
        imgs = torch.cat(imgs, dim=0)
        real = torch.cat(real, dim=0)
        sss = torch.cat([imgs, real], dim=0)

        fid = calculate_fid_modify(imgs[0, 0, :, :].detach().numpy(), real[0, 0, :, :].detach().numpy())

        # Linearly calibrated models (LPIPS)
        loss_fn = lpips.LPIPS(net='alex', spatial=True)  # Can also set net = 'squeeze' or 'vgg'
        # loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'
        imgs = lpips.tensor2im(imgs)
        imgs = lpips.im2tensor(imgs)
        real.data = torch.clamp(real.data, -1, 1)
        real = lpips.tensor2im(real)
        real = lpips.im2tensor(real)
        similarity = torch.cosine_similarity(imgs, real, dim=1).mean()
        # real.data = torch.clamp(real.data, -1, 1)
        dist = loss_fn(imgs, real)
        dist = dist.mean()

        print('num-iter-------------', n_iter+1)
        print('fid-------------(%.1f)' % fid)
        # print('lpips-------------', dist)
        print('cosine_similarity-------------(%.2f)' % similarity)
        print('lpips-------------(%.2f)' % dist)
        vutils.save_image(sss, "%s/iter_%d.jpg" % (saved_image_folder, n_iter), range=(-1, 1), nrows = 12, normalize=True)
        del imgs
    net_g.train()
    return fid, similarity, dist, n_iter

def test(net_g, net_d, max_iteration):
    global fid, similarity, dist
    fid = 0
    similarity =0
    dist = 0
    from model_ours import num_rounds
    global saved_model_folder, saved_image_folder


    num_rounds = num_rounds()

    if max_iteration != None:
        for i in range(num_rounds):
            # max_iteration = max_iteration * (i + 1)
            path_image = saved_image_folder + '/iter_%d.jpg' % (max_iteration*(i + 1))
            # path_image = saved_image_folder + '/iter_%d.jpg' % (max_iteration * i)
            if os.path.exists(path_image) == False:
                fid, similarity, dist, num_rounds = save_image(net_g, dataloader_A_fixed, saved_image_folder, max_iteration*(i + 1))
                # fid, similarity, dist, num_rounds = save_image(net_g, dataloader_A_fixed, saved_image_folder, max_iteration * i)
                break
    return fid, similarity, dist, num_rounds

def train(net_g, net_d, max_iteration, proximal_mu):
    print('training begin ... ')
    titles = ['gram', 'cos_skt', 'net_g', 'loss_clip_skt', 'loss_clip_rgb','ssim']
    losses = {title: 0.0 for title in titles}

    real_style = next(dataloader_B)[0].to(device)
    sf_1, sf_2, sf_3, sf_4, feat_skt = vgg(real_style)
    mean_1, std_1 = torch.mean(sf_1, dim=[0, 2, 3], keepdim=True), \
                    torch.std(sf_1, dim=[0, 2, 3], keepdim=True)
    mean_2, std_2 = torch.mean(sf_2, dim=[0, 2, 3], keepdim=True), \
                    torch.std(sf_2, dim=[0, 2, 3], keepdim=True)
    mean_3, std_3 = torch.mean(sf_3, dim=[0, 2, 3], keepdim=True), \
                    torch.std(sf_3, dim=[0, 2, 3], keepdim=True)
    mean_4, std_4 = torch.mean(sf_4, dim=[0, 2, 3], keepdim=True), \
                    torch.std(sf_4, dim=[0, 2, 3], keepdim=True)

    for n_iter in tqdm.tqdm(range(max_iteration + 1)):

        real_content = next(dataloader_A)[0].to(device)
        real_style = next(dataloader_B)[0].to(device)
        rgb_data, skt_data = real_content, real_style

        # 1. extract feature
        # 1.1 pre-feature (vgg)
        rf_1, rf_2, rf_3, rf_4, feat_rgb = vgg(rgb_data)
        sf_1, sf_2, sf_3, sf_4, feat_skt = vgg(skt_data)
        # (B, 64, 256, 256)
        # (B, 128, 128, 128)
        # (B, 256, 64, 64)
        # (B, 512, 32, 32)

        # visualize_features(rf_1)
        # visualize_features(sf_1)

        # 1.2 sketch generate
        rf_4, rf_3, rf_2, rf_1 = AdaIN_GAI(rf_4, mean_4, std_4), AdaIN_GAI(rf_3, mean_3, std_3),\
                                 AdaIN_GAI(rf_2, mean_2, std_2), AdaIN_GAI(rf_1, mean_1, std_1)
        skt_gen = net_g(rf_4, rf_3, rf_2, rf_1)

        # 1.3 generate skt feature
        gf_1, gf_2, gf_3, gf_4, feat_gen = vgg(skt_gen)

        # 2 calculate losses
        # multi_visualize([rf_4, sf_4, AdaINN(rf_4, sf_4), gf_4])

        # 2.1 cosine loss (skt_gen | skt)
        loss_cos_skt = 0.5 * loss_for_cos(feat_skt, feat_gen)

        # 2.2 clip loss (skt_gen | skt) (skt_gen | rgb)
        skt_feat, gen_feat, rgb_layer, skt_layer = clip_feature(clip_model, rgb_data, skt_data, skt_gen)
        loss_clip_skt = loss_for_cos(skt_feat, gen_feat)
        loss_clip_rgb = 100 * F.mse_loss(rgb_layer, skt_layer)

        # 2.3 multi-scale gram matrix
        loss_gram = 100 * (F.mse_loss(gram_matrix(sf_4), gram_matrix(gf_4)) +
                           F.mse_loss(gram_matrix(sf_3), gram_matrix(gf_3)) +
                           F.mse_loss(gram_matrix(sf_2), gram_matrix(gf_2)) +
                           F.mse_loss(gram_matrix(sf_1), gram_matrix(gf_1)))

        # 3. train Discriminator
        net_d.zero_grad()
        # 3.1. training on real data and fake data
        gram_sf_4 = avg_pool(gram_matrix(sf_4))
        gram_sf_3 = avg_pool(gram_matrix(sf_3))
        gram_sf_2 = avg_pool(gram_matrix(sf_2))
        gram_sf_1 = avg_pool(gram_matrix(sf_1))  # (B, 64, 14, 14)
        skt_data_sample = torch.cat([gram_sf_1, gram_sf_2, gram_sf_3, gram_sf_4], dim=1)

        gram_gf_4 = avg_pool(gram_matrix(gf_4))
        gram_gf_3 = avg_pool(gram_matrix(gf_3))
        gram_gf_2 = avg_pool(gram_matrix(gf_2))
        gram_gf_1 = avg_pool(gram_matrix(gf_1))  # (B, 64, 14, 14)
        skt_gen_sample = torch.cat([gram_gf_1, gram_gf_2, gram_gf_3, gram_gf_4], dim=1)

        train_d(net_d, skt_data_sample.detach(), label="real")
        train_d(net_d, skt_gen_sample.detach(), label="fake")
        optD.step()

        # 4. train Generator
        net_g.zero_grad()
        # 4.1. train G as real image
        pred_gs = net_d(skt_gen_sample)
        loss_g = -pred_gs.mean()

        feddecorr = FedDecorrLoss()
        # loss_feddecorr_g = feddecorr(fake_img)
        loss_feddecorr_d_style = feddecorr(pred_gs.unsqueeze(0))

        # 5. backward
        # loss = loss_gram + loss_cos_skt + loss_g + loss_clip_skt + loss_clip_rgb + args.feddecorr_coef * loss_feddecorr_d_style

        global_params = [val.detach().clone() for val in net_d.parameters()]
        proximal_term = 0.0
        for local_weights, global_weights in zip(net_d.parameters(), global_params):
            proximal_term += (local_weights - global_weights).norm(2)

        loss = loss_gram + loss_cos_skt + loss_g + loss_clip_skt + loss_clip_rgb + (proximal_mu / 2) * proximal_term

        loss.backward()
        optG.step()

        ssim_val = calculate_ssim(skt_data, skt_gen)

        # 6. logging
        loss_values = [loss_gram, loss_cos_skt.item(), loss_g.item(),
                        loss_clip_skt, loss_clip_rgb, ssim_val]
        for i, term in enumerate(titles):
            losses[term] += loss_values[i]

        # if (n_iter + 1) % (100) == 0:
        #     fid, similarity, dist, n_iter = save_image(net_g, dataloader_A_fixed, saved_image_folder, n_iter)
        #     wandb.log({"FID": fid, "cosine_similarity": similarity, "LPIPS": dist}, step=n_iter + 1)
        #
        #     if fid <= best_fid:
        #         best_fid[index] = fid
        #         best_epoch[index] = n_iter
        #         try:
        #             model_dict = {'g': net_g.state_dict(), 'ds': net_d.state_dict()}
        #             D_dict = {'ds': net_d.state_dict()}
        #             torch.save(model_dict, os.path.join(saved_model_folder, 'model_best.pth'))
        #             vutils.save_image(torch.cat([rgb_data, skt_gen, skt_data], dim=0),
        #                               os.path.join(saved_image_folder, 'iter_%d.png' % (n_iter)),
        #                               nrow=8, range=(-1, 1), normalize=True)
        #             torch.save(D_dict, os.path.join(saved_model_folder, 'D_best.pth'))
        #             # opt_dict = {'g': optG.state_dict(), 'ds': optDS.state_dict()}
        #             # torch.save(opt_dict, os.path.join(saved_model_folder, '%d_opt.pth' % (n_iter)))
        #         except:
        #             print("models not properly saved")
        # if idx == 0:
        # vutils.save_image(torch.cat([rgb_data, skt_gen, skt_data], dim=0),
        #                   os.path.join(saved_image_folder, 'iter_%d.png' % (n_iter)),
        #                   nrow=8, range=(-1, 1), normalize=True)
            # torch.save(net_g.state_dict(), os.path.join(saved_model_folder, '%d.pth' % (n_iter)))

    log_line = "Epoch[{}/{}]  ".format(n_iter, max_iteration)
    for key, value in losses.items():
        log_line += "%s: %.5f  " % (key, value)
        losses[key] = 0
    print(log_line)




class FedBNClient(fl.client.NumPyClient):

    def __init__(
        self,
        net_g: Generator,
        net_d: Discriminator,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.net_g = net_g
        self.net_d = net_d
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        # self.net_g.train()
        self.net_d.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding parameters of BN layers when using FedBN
            # params_net_g = [val.cpu().numpy() for name, val in net_g.state_dict().items() if "bn" not in name]
            params_net_d = [val.cpu().numpy() for name, val in net_d.state_dict().items() if "bn" not in name]
            parameters = params_net_d
            return parameters
        else:
            # Return model parameters as a list of NumPy ndarrays
            # params_net_g = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            params_net_d = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            parameters = params_net_d
            return parameters
            # return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        # self.net_g.train()
        self.net_d.train()
        if USE_FEDBN:
            # keys_net_g = [k for k in self.net_g.state_dict().keys() if "bn" not in k]
            # params_dict_net_g = zip(keys_net_g, parameters[0])
            # state_dict_net_g = OrderedDict({k: torch.tensor(v) for k, v in params_dict_net_g})
            # self.net_g.load_state_dict(state_dict_net_g, strict=False)

            keys_net_d = [k for k in self.net_d.state_dict().keys() if "bn" not in k]
            params_dict_net_d = zip(keys_net_d, parameters)
            state_dict_net_d = OrderedDict({k: torch.tensor(v) for k, v in params_dict_net_d})
            self.net_d.load_state_dict(state_dict_net_d, strict=False)
        else:
            # params_dict_net_g = zip(self.net_g.state_dict().keys(), parameters[0])
            # state_dict_net_g = OrderedDict({k: torch.tensor(v) for k, v in params_dict_net_g})
            # self.net_g.load_state_dict(state_dict_net_g, strict=True)

            params_dict_net_d = zip(self.net_d.state_dict().keys(), parameters)
            state_dict_net_d = OrderedDict({k: torch.tensor(v) for k, v in params_dict_net_d})
            self.net_d.load_state_dict(state_dict_net_d, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        global isEpoch_tr
        self.set_parameters(parameters)
        if isEpoch_tr:
            train(net_g, net_d, 0)
            isEpoch_tr = False
        else:
            train(net_g, net_d, max_iteration)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:

        # Set model parameters, evaluate model on local test dataset, return result
        global isEpoch_te
        self.set_parameters(parameters)
        if isEpoch_te:
            fid_0, similarity_0, dist_0, _ = test(net_g, net_d, 0)
            FID = fid_0
            Simlarity = similarity_0
            Dist = dist_0
            Num_rounds = 0
            isEpoch_te = False
        else:
            fid, similarity, dist, num_rounds = test(net_g, net_d, max_iteration)
            FID = fid
            Simlarity = similarity
            Dist = dist
            Num_rounds = num_rounds
        wandb.log({"FID": FID, "cosine_similarity": Simlarity, "LPIPS": Dist}, step=Num_rounds)
        return float(FID), self.num_examples["testset"], { "LPIPS": float(Dist)}
        # return float(FID), float(Simlarity), float(Dist)

class FedAvgClient(fl.client.NumPyClient):
    def __init__(
        self,
        net_g: Generator,
        net_d: Discriminator,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
    ) -> None:
        self.net_g = net_g
        self.net_d = net_d
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples


    def get_parameters(self, config) -> List[np.ndarray]:
        # self.net_g.train()
        self.net_d.train()
        params_net_d = [val.cpu().numpy() for name, val in net_d.state_dict().items()]
        parameters = params_net_d
        return parameters

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        # self.net_g.train()
        self.net_d.train()
        params_dict_net_d = zip(self.net_d.state_dict().keys(), parameters)
        state_dict_net_d = OrderedDict({k: torch.tensor(v) for k, v in params_dict_net_d})
        self.net_d.load_state_dict(state_dict_net_d, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        global isEpoch_tr
        self.set_parameters(parameters)
        if isEpoch_tr:
            train(net_g, net_d, 0)
            isEpoch_tr = False
        else:
            train(net_g, net_d, max_iteration)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:

        # Set model parameters, evaluate model on local test dataset, return result
        global isEpoch_te
        self.set_parameters(parameters)
        if isEpoch_te:
            fid_0, similarity_0, dist_0, _ = test(net_g, net_d, 0)
            FID = fid_0
            Simlarity = similarity_0
            Dist = dist_0
            Num_rounds = 0
            isEpoch_te = False
        else:
            fid, similarity, dist, num_rounds = test(net_g, net_d, max_iteration)
            FID = fid
            Simlarity = similarity
            Dist = dist
            Num_rounds = num_rounds
        wandb.log({"FID": FID, "cosine_similarity": Simlarity, "LPIPS": Dist}, step=Num_rounds)
        return float(FID), self.num_examples["testset"], {"cosine_similarity": float(Simlarity), "LPIPS": float(Dist)}

class FedProxClient(fl.client.NumPyClient):

    def __init__(
        self,
        net_g: Generator,
        net_d: Discriminator,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict,
        proximal_mu: float
    ) -> None:
        self.net_g = net_g
        self.net_d = net_d
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.proximal_mu = proximal_mu

    def get_parameters(self, config) -> List[np.ndarray]:
        # self.net_g.train()
        self.net_d.train()
        params_net_d = [val.cpu().numpy() for name, val in net_d.state_dict().items()]
        parameters = params_net_d
        return parameters

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        # self.net_g.train()
        self.net_d.train()
        params_dict_net_d = zip(self.net_d.state_dict().keys(), parameters)
        state_dict_net_d = OrderedDict({k: torch.tensor(v) for k, v in params_dict_net_d})
        self.net_d.load_state_dict(state_dict_net_d, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        global isEpoch_tr
        self.set_parameters(parameters)
        if isEpoch_tr:
            train(net_g, net_d, 0, proximal_mu)
            isEpoch_tr = False
        else:
            train(net_g, net_d, max_iteration, proximal_mu)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # wandb.init(project="FedAdam_ZGX", config=args)
        # Set model parameters, evaluate model on local test dataset, return result
        global isEpoch_te
        self.set_parameters(parameters)
        if isEpoch_te:
            fid_0, similarity_0, dist_0, _ = test(net_g, net_d, 0)
            FID = fid_0
            Simlarity = similarity_0
            Dist = dist_0
            Num_rounds = 0
            isEpoch_te = False
        else:
            fid, similarity, dist, num_rounds = test(net_g, net_d, max_iteration)
            FID = fid
            Simlarity = similarity
            Dist = dist
            Num_rounds = num_rounds
        wandb.log({"FID": FID, "cosine_similarity": Simlarity, "LPIPS": Dist}, step=Num_rounds)
        return float(FID), self.num_examples["testset"], {"cosine_similarity": float(Simlarity), "LPIPS": float(Dist)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Style transfer GAN, during training, the model will learn to take a image from one specific catagory and transform it into another style domain')
    print(os.path.join(os.getcwd(), "art-landscape-rgb-512"))
    # patha="RGB/"
    # pathb="Sketch/"


    patha = "./D4/RGB"
    pathb = "./D4/sketch"
    parser.add_argument('--path_a', type=str, default=patha,
                        help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--path_b', type=str, default=pathb,
                        help='path of target dataset, should be a folder that has one or many sub image folders inside')

    parser.add_argument('--im_size', type=int, default=256, help='resolution of the generated images')
    parser.add_argument('--trial_name', type=str, default="D4_ours", help='a brief description of the training trial')
    # parser.add_argument('--gpu_id', type=int, default=0, help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate, default is 2e-4, usually dont need to change it, you can try make it smaller, such as 1e-4')
    parser.add_argument('--batch_size', type=int, default=1, help='how many images to train together at one iteration')
    parser.add_argument('--total_iter', type=int, default=11,
                        help='how many iterations to train in total, the value is in assumption that init step is 1')
    parser.add_argument('--clip', type=str, default='ViT-B/16', help='Use clip pre-trained model type')
    parser.add_argument('--feddecorr_coef', type=float, default=0.1, help='coefficient of FedDecorr')
    parser.add_argument('--proximal_mu', type=float, default=1.0, help='coefficient of proximal term')
    # parser.add_argument('--isEpoch', default=True, type=bool,
    #                     help='Output the three loss index values of the initial model')

    pathc = "./train_results/D4_ours/models/model_best.pth"
    parser.add_argument('--checkpoint', default=pathc, type=str, help='specify the path of the pre-trained model')

    args = parser.parse_args()

    print(str(args))

    # isEpoch = args.isEpoch
    trial_name = args.trial_name
    data_root_A = args.path_a
    data_root_B = args.path_b
    proximal_mu = args.proximal_mu
    # mse_weight = args.mse_weight
    # gram_weight = args.gram_weight
    max_iteration = args.total_iter
    # device = torch.device("cuda:%d"%(args.gpu_id))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(torch.cuda.is_available())
    # print(torch.cuda.current_device())
    im_size = args.im_size
    if im_size == 128:
        base = 4
    elif im_size == 256:
        base = 8
    elif im_size == 512:
        base = 16
    if im_size not in [128, 256, 512]:
        print("the size must be in [128, 256, 512]")

    save_folder = './'

    # Load Networks
    vgg = VGGSimple()
    vgg.eval().to(device)
    for p in vgg.parameters():
        p.requires_grad = False

    clip_model, _ = clip.load(args.clip, device=device, jit=False)
    for param in clip_model.parameters():
        param.requires_grad = False

    avg_pool = Adaptive_pool(channel_out=64, hw_out=14)

    net_g = Generator(nfc=256, ch_out=3)
    net_g.to(device)
    optG = optim.SGD(net_g.parameters(), lr=args.lr, momentum=0.9)

    net_d = Discriminator(nfc=64 * 4)
    net_d.to(device)
    optD = optim.SGD(net_d.parameters(), lr=args.lr, momentum=0.9)

    dataset_A = Dataset.ImageFolder(root=data_root_A, transform=trans_maker(args.im_size))
    dataloader_A_fixed = DataLoader(dataset_A, 12, shuffle=False, num_workers=0)
    dataloader_A = iter(DataLoader(dataset_A, args.batch_size, shuffle=False, \
                                   sampler=InfiniteSamplerWrapper(dataset_A), num_workers=0, pin_memory=False))

    dataset_B = Dataset.ImageFolder(root=data_root_B, transform=trans_maker(args.im_size))
    dataloader_B = iter(DataLoader(dataset_B, args.batch_size, shuffle=False, \
                                   sampler=InfiniteSamplerWrapper(dataset_B), num_workers=0, pin_memory=False))



    if args.checkpoint != 'None':
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        net_g.load_state_dict(checkpoint['g'])
        net_d.load_state_dict(checkpoint['ds'])
        print("saved model loaded")

    net_d.to(device)
    net_g.to(device)



    if args.checkpoint != 'None':
        opt_path = args.checkpoint.replace("_model.pth", "_opt.pth")
        try:
            opt_weights = torch.load(opt_path, map_location=lambda a, b: a)
            optG.load_state_dict(opt_weights['g'])
            optD.load_state_dict(opt_weights['ds'])
            print("saved optimizer loaded")
        except:
            print(
                "no optimizer weights detected, resuming a training without optimizer weights may not let the model converge as desired")
            pass

    # train(net_g, net_d, max_iteration)

    """Load data, start CifarClient."""

    # Load data
    trainloader, testloader, num_examples = load_data()
    saved_model_folder, saved_image_folder = creat_folder(save_folder, trial_name)
    # Start client
    # wandb.init(project="FedGAI", config=args)

    wandb.init(project="FedGAI_st_baseline", config=args, name="D4_ours")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.run.name += f"_{current_time}"

    # client = FedAvgClient(net_g, net_d, trainloader, testloader, num_examples)
    client = FedProxClient(net_g, net_d, trainloader, testloader, num_examples, proximal_mu)

    # fl.client.start_numpy_client(server_address="10.188.141.30:8080", client=client, grpc_max_message_length = 1500*1024*1024)
    fl.client.start_numpy_client(
                                     # server_address="10.188.141.30:8080",
                                     server_address="Your IP address:8080",
                                     client=client,
                                     grpc_max_message_length = 1500*1024*1024,
                                     # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
                                     )
    wandb.finish()