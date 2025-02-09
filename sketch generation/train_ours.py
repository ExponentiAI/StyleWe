import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model_ours import Generator, VGGSimple, Discriminator, train_d, Adaptive_pool, AdaIN_GAI
from operation_ours import LoadMyDataset, LoadSingleDataset, creat_folder, loss_for_cos, gram_matrix, calculate_ssim, \
    clip_feature, visualize_features, multi_visualize, trans_maker, InfiniteSamplerWrapper
import tqdm
import random
import clip
import numpy as np
import torchvision.utils as vutils
import torchvision.datasets as Dataset
import argparse
import gc
from metrics import calculate_fid_modify
import lpips
import wandb
import datetime
from thop import profile


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
        vutils.save_image(sss, "%s/iter_%d.jpg" % (saved_image_folder, n_iter), range=(-1, 1), normalize=True)
        del imgs
    net_g.train()
    return fid, similarity, dist, n_iter



def train(args):
    print('training begin ... ')
    titles = ['gram', 'cos_skt', 'net_g', 'loss_clip_skt', 'loss_clip_rgb','ssim']
    losses = {title: 0.0 for title in titles}

    saved_model_folder, saved_image_folder = creat_folder(save_folder, trial_name)

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

    # FID
    best_fid = [float('inf')]
    best_epoch = [0]
    index = 0

    fid, similarity, dist, n_iter = save_image(net_g, dataloader_A_fixed, saved_image_folder, 0)
    wandb.log({"FID": fid, "cosine_similarity": similarity, "LPIPS": dist}, step=n_iter)

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
        flops_stu, params_stu = profile(net_g, (rf_4, rf_3, rf_2, rf_1,))
        print('flops_stu: {:.2f} GFLOPs, params_stu: {:.2f} MB'.format(flops_stu / 1e9, params_stu / 1e6))


        # 1.3 generate skt feature
        gf_1, gf_2, gf_3, gf_4, feat_gen = vgg(skt_gen)

        # 2 calculate losses
        # multi_visualize([rf_4, sf_4, AdaINN(rf_4, sf_4), gf_4])

        # 2.1 cosine loss (skt_gen | skt)
        loss_cos_skt = 0.5 * loss_for_cos(feat_skt, feat_gen)

        # 2.2 clip loss (skt_gen | skt) (skt_gen | rgb)
        skt_feat, gen_feat, rgb_layer, skt_layer = clip_feature(clip_model, rgb_data, skt_data, skt_gen)
        # flops_clip, params_clip = profile(clip_model, (clip_model.encode_image(skt_data), clip_model.encode_image(skt_gen),))
        # print('flops_clip: {:.2f} GFLOPs, params_clip: {:.2f} MB'.format(flops_clip / 1e9, params_clip / 1e6))

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

        # 5. backward
        loss = loss_gram + loss_cos_skt + loss_g + loss_clip_skt + loss_clip_rgb

        loss.backward()
        optG.step()

        ssim_val = calculate_ssim(skt_data, skt_gen)

        # 6. logging
        loss_values = [loss_gram, loss_cos_skt.item(), loss_g.item(),
                        loss_clip_skt, loss_clip_rgb, ssim_val]
        for i, term in enumerate(titles):
            losses[term] += loss_values[i]

        if (n_iter + 1) % (100) == 0:
            fid, similarity, dist, n_iter = save_image(net_g, dataloader_A_fixed, saved_image_folder, n_iter)
            wandb.log({"FID": fid, "cosine_similarity": similarity, "LPIPS": dist}, step=n_iter + 1)

            # if fid <= best_fid:
            #     best_fid[index] = fid
            #     best_epoch[index] = n_iter
            try:
                model_dict = {'g': net_g.state_dict(), 'ds': net_d.state_dict()}
                D_dict = {'ds': net_d.state_dict()}
                torch.save(model_dict, os.path.join(saved_model_folder, 'model_%d.pth' % (n_iter)))
                vutils.save_image(torch.cat([rgb_data, skt_gen, skt_data], dim=0),
                                  os.path.join(saved_image_folder, 'iter_%d.png' % (n_iter)),
                                  nrow=8, range=(-1, 1), normalize=True)
                torch.save(D_dict, os.path.join(saved_model_folder, 'D_%d.pth' % (n_iter)))
                # opt_dict = {'g': optG.state_dict(), 'ds': optDS.state_dict()}
                # torch.save(opt_dict, os.path.join(saved_model_folder, '%d_opt.pth' % (n_iter)))
            except:
                print("models not properly saved")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sketch Generator')
    parser.add_argument('--path_a', type=str, default='./why/RGB',
                        help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--path_b', type=str, default='./why/sketch',
                        help='path of target dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--trial_name', type=str, default="WHY_ours", help='a brief description of the training trial')
    parser.add_argument('--seed', type=int, default=24, help='seed')
    parser.add_argument('--im_size', type=int, default=256, help='size of generated images')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--total_iter', type=int, default=7000, help='the iterations to train in total')
    parser.add_argument('--clip', type=str, default='ViT-B/16', help='Use clip pre-trained model type')
    args = parser.parse_args()
    print(str(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('\n[INFO] Setting SEED: ' + str(args.seed))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_root_A = args.path_a
    data_root_B = args.path_b
    dataset_A = Dataset.ImageFolder(root=data_root_A, transform=trans_maker(args.im_size))
    dataloader_A_fixed = DataLoader(dataset_A, 8, shuffle=False, num_workers=0)
    dataloader_A = iter(DataLoader(dataset_A, args.batch_size, shuffle=False, \
                                   sampler=InfiniteSamplerWrapper(dataset_A), num_workers=4, pin_memory=False))

    dataset_B = Dataset.ImageFolder(root=data_root_B, transform=trans_maker(args.im_size))
    dataloader_B = iter(DataLoader(dataset_B, args.batch_size, shuffle=False, \
                                   sampler=InfiniteSamplerWrapper(dataset_B), num_workers=0, pin_memory=False))

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

    im_size = args.im_size
    if im_size == 128:
        base = 4
    elif im_size == 256:
        base = 8
    elif im_size == 512:
        base = 16
    if im_size not in [128, 256, 512]:
        print("the size must be in [128, 256, 512]")

    max_iteration = args.total_iter
    trial_name = args.trial_name
    save_folder = './'

    wandb.init(project="1.I2S_Gen", config=args, name="WHY_ours")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb.run.name += f"_{current_time}"

    train(args)
    wandb.finish()