from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map
import json

import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time
from itertools import cycle
from model.ssl.CTmodel import CCT
from model.ssl.encoder import Deeplabv3plusEncoder, PSPNetEncoder, DeepLabV2Encoder
from model.ssl.decoders import *
from Tools.utils.losses import *
from model.ssl.metrics import AverageMeter
from Tools.CRF import CRF



MODE = None


def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes', 'sodl', 'ir', 'whdld'], default='pascal')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)

    parser.add_argument('--save-path', type=str, required=True)

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', dest='plus', default=False, action='store_true',
                        help='whether to use ST++')

    args = parser.parse_args()
    return args

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(args, config):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=4 if args.dataset == 'cityscapes' else 4,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    # <====================== 第一阶段一致性正则化训练 ======================>
    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (8))

    global MODE

    #设置数据增强模式 train:弱数据增强
    MODE = 'train'
    unlabeledtrain_id_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/dataset/splits/whdld/1_4/split_0/unlabeled_train.txt'

    trainset_sup = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    # trainset_sup.ids = 2 * trainset_sup.ids if len(trainset_sup.ids) < 200 else trainset_sup.ids
    trainloader_sup = DataLoader(trainset_sup, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=6, drop_last=True)

    trainset_unsup = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, unlabeledtrain_id_path)
    trainloader_unsup = DataLoader(trainset_unsup, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=6, drop_last=True)



    num_classes = 6
    iter_per_epoch = len(trainloader_sup)

    # SUPERVISED LOSS
    if config['model']['sup_loss'] == 'CE':
        sup_loss = CE_loss
    elif config['model']['sup_loss'] == 'FL':
        alpha = get_alpha(trainloader_sup) # calculare class occurences
        sup_loss = FocalLoss(apply_nonlin = softmax_helper, ignore_index = config['ignore_index'], alpha = alpha, gamma = 2, smooth = 1e-5)
    else:
        sup_loss = abCE_loss(iters_per_epoch=iter_per_epoch, epochs=args.epochs,
                                num_classes=num_classes)

    # MODEL
    rampup_ends = int(config['ramp_up'] * args.epochs)
    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=len(trainloader_unsup),
                                        rampup_ends=rampup_ends)

    ctmodel = CCT(num_classes=num_classes, conf=config['model'],
    						sup_loss=sup_loss, cons_w_unsup=cons_w_unsup,
    						weakly_loss_w=config['weakly_loss_w'], use_weak_lables=config['use_weak_lables'],
                            ignore_index=255)

    print('\nmodel Params: %.1fM' % count_params(ctmodel))

    lr = config['optimizer']['args']['lr'] / 10

    trainable_params = [{'params': filter(lambda p: p.requires_grad, ctmodel.get_other_params())},
                        {'params': filter(lambda p: p.requires_grad, ctmodel.get_backbone_params()),
                         'lr': lr}]

    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)

    # 训练模式选择
    # best_name = semi_train_start(ctmodel,  trainloader_sup, trainloader_unsup, valloader, criterion, optimizer, args, 'stage1', lr)

    # model, optimizer = init_basic_elems(args)
    # #手动模型载入
    # best_name = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/exprements/ir/id=64/deeplabv3plus_resnet50_57.16.pth'
    # best_model_path = best_name
    #
    # best_model = model
    # best_model.module.load_state_dict(torch.load(best_model_path))
    #
    # print('\n\n\n================> Total stage 2/8: 在无标签数据中划分可靠数据和不可靠数据')
    #
    # dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    #
    # select_reliable(best_model, dataloader, args, istest=True)
    #
    #
    print('\n\n\n================> Total stage 3/8: 对可靠数据使用教师模型生成伪标签')

    # cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    # cur_unlabeled_id_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/reliable_ids/whdld/1_16/split_0/reliable_ids.txt'
    # dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    #
    # label(best_model, dataloader, args, istest=True)

    # print('\n\n\n================> Total stage 4/8: 使用条件随机场+空洞修补来精细化伪标签')
    #
    # original_image_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/IR/JPEGImages/'
    # predicted_image_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/pseudo_masks/ir/1_4/split_0'
    # CRF_Fill_image_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/pseudo_masks/ir/1_4/split_0_mask/'
    # CRF(original_image_path, predicted_image_path, CRF_Fill_image_path)
    #
    # print('\n\n\n================> Total stage 5/8: 在有标签数据和可靠无标签数据上重新训练学生模型')
    #
    # MODE = 'semi_train'
    # cur_unlabeled_id_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/reliable_ids/whdld/1_16/split_0/reliable_ids.txt'
    # pseudo_mask_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/pseudo_masks/whdld/1_16/split_0'
    # trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
    #                        args.labeled_id_path, cur_unlabeled_id_path, pseudo_mask_path)
    # trainloader = DataLoader(trainset, batch_size=16, shuffle=True,
    #                          pin_memory=True, num_workers=12, drop_last=True)
    # trainloader_unsup = None
    #
    # semi_optimizer = optimizer
    # lr = config['optimizer']['args']['lr'] / 10
    #
    # print('\nmodel Params: %.1fM' % count_params(ctmodel))
    #
    # best_name = train_start(ctmodel, trainloader, trainloader_unsup, valloader, criterion, semi_optimizer, args, 'stage2', lr)
    #
    # 手动模型载入
    best_name = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/exprements/sodl/id=4/deeplabv3plus_resnet50_77.12.pth'
    best_model_path = best_name
    best_model = ctmodel
    best_model.load_state_dict(torch.load(best_model_path))

    print('\n\n\n================> Total stage 6/8: 使用学生模型对不可靠无标签数据生成伪标签')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    cur_unlabeled_id_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/dataset/splits/whdld/1_16/split_0/unlabeled.txt'
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args, istest=True)

    # print('\n\n\n================> Total stage 7/8: 使用条件随机场+空洞修补来精细化伪标签')
    #
    # original_image_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/SODL/JPEGImages/'
    # predicted_image_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/pseudo_masks/sodl/1_4/split_0'
    # CRF_Fill_image_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/pseudo_masks/sodl/1_4/split_0_mask/'
    # CRF(original_image_path, predicted_image_path, CRF_Fill_image_path)

    print('\n\n\n================> Total stage 8/8: The 2nd stage re-training on labeled and all unlabeled images')

    MODE = 'semi_train'
    pseudo_mask_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/pseudo_masks/whdld/1_16/split_0'
    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True,
                             pin_memory=True, num_workers=12, drop_last=True)
    trainloader_unsup = None

    semi_model = ctmodel
    semi_optimizer = optimizer
    print('\nmodel Params: %.1fM' % count_params(ctmodel))

    train_start(semi_model, trainloader, trainloader_unsup, valloader, criterion, semi_optimizer, args, 'stage3', lr)








def init_basic_elems(args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone, 21 if args.dataset == 'pascal' else 6)

    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
        head_lr_multiple = 1.0

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()


    return model, optimizer


def _get_available_devices( n_gpu):
    sys_gpu = torch.cuda.device_count()
    if sys_gpu == 0:
        print('No GPUs detected, using the CPU')
        n_gpu = 0
    elif n_gpu > sys_gpu:
        print(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
        n_gpu = sys_gpu

    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    available_gpus = list(range(n_gpu))

    return device, available_gpus


def update_losses(cur_losses, loss_sup, loss_unsup, isSUP):
    if isSUP:
        loss_sup = cur_losses['loss_sup'].mean().item()
        return loss_sup
    loss_sup = cur_losses['loss_sup'].mean().item()
    loss_unsup = cur_losses['loss_unsup'].mean().item()
    return loss_sup, loss_unsup



def semi_train_start(model, trainloader_sup, trainloader_unsup, valloader, criterion, optimizer, args, flag, lr):
    # Training
    t_start = time.time()
    log = "./logs/"
    if not os.path.exists(log + flag):
        os.makedirs(log + flag)
    logdir = log + flag
    writer = SummaryWriter(logdir)

    loss_sup = AverageMeter()
    loss_unsup = AverageMeter()


    iters = 0
    # total_iters = len(trainloader_sup) * args.epochs

    previous_best = 0.0

    global MODE, checkpoints, best_model, best_name

    if MODE == 'train':
        checkpoints = []

    device, availble_gpus = _get_available_devices(1)
    model = torch.nn.DataParallel(model, device_ids=availble_gpus)
    model.to(device)

    #---- 半监督 ----#
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        # model_teacher.train()

        total_loss = 0.0

        dataloader = iter(zip(cycle(trainloader_sup), trainloader_unsup))
        tbar = tqdm(range(len(trainloader_sup)), ncols=135)

        for batch_idx in tbar:

            (input_l, target_l), (input_ul, target_ul) = next(dataloader)
            input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
            input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)
            optimizer.zero_grad()

            loss, cur_losses, outputs = model(x_l=input_l, target_l=target_l, x_ul=input_ul,curr_iter=batch_idx,
                                              target_ul=target_ul, epoch=epoch - 1)

            loss = loss.mean()
            loss.backward()
            optimizer.step()

            loss_sup, loss_unsup = update_losses(cur_losses, loss_sup, loss_unsup, isSUP=False)
            #
            # lr = lr * (1 - epoch / args.epochs) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr
            # optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0
            tbar.set_description('T ({}) | Ls {:.2f} Lu {:.2f}|'.format(epoch, loss_sup, loss_unsup))


        writer.add_scalar('train/total_loss_epoch', (total_loss / (iters + 1)), epoch)
        metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 6)

        model.eval()
        tbar = tqdm(valloader)

        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model(x_l=img)
                pred = torch.argmax(pred, dim=1)

                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[1]
                classIOU = metric.evaluate()[0]


                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
                writer.add_scalar('test/mIOU_epoch', mIOU, epoch)

        mIOU *= 100.0

        print('mIOU: %.2f' % (mIOU))
        cIOU = classIOU.tolist()
        print('class1IOU: %.2f' % (cIOU[0] * 100.0))
        print('class2IOU: %.2f' % (cIOU[1] * 100.0))
        print('class3IOU: %.2f' % (cIOU[2] * 100.0))
        print('class4IOU: %.2f' % (cIOU[3] * 100.0))
        print('class5IOU: %.2f' % (cIOU[4] * 100.0))
        print('class6IOU: %.2f' % (cIOU[5] * 100.0))

        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

        best_name = os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best))
        print(best_name)
            # best_model = deepcopy(model)
        #
        # if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
        #     checkpoints.append(deepcopy(model))
    #
    # if MODE == 'train':
    #     return best_model, checkpoints

    return best_name

def train_start(model, trainloader_sup, trainloader_unsup, valloader, criterion, optimizer, args, flag, lr):
    # Training
    t_start = time.time()
    log = "./logs/"
    if not os.path.exists(log + flag):
        os.makedirs(log + flag)
    logdir = log + flag
    writer = SummaryWriter(logdir)

    loss_sup = AverageMeter()
    loss_unsup = AverageMeter()


    iters = 0
    # total_iters = len(trainloader_sup) * args.epochs

    previous_best = 0.0

    global MODE, checkpoints, best_model, best_name

    if MODE == 'train':
        checkpoints = []

    device, availble_gpus = _get_available_devices(1)
    model = torch.nn.DataParallel(model, device_ids=availble_gpus)
    model.to(device)

    ## ----- 全监督 ---- ##
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()

        total_loss = 0.0

        dataloader = iter(trainloader_sup)
        tbar = tqdm(range(len(trainloader_sup)), ncols=135)

        for batch_idx in tbar:
            input_l, target_l = next(dataloader)
            input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
            optimizer.zero_grad()
            loss, cur_losses, outputs = model(x_l=input_l, target_l=target_l, x_ul=None, curr_iter=batch_idx,
                                              target_ul=None, epoch=epoch - 1)

            loss = loss.mean()
            loss.backward()
            optimizer.step()

            loss_sup = update_losses(cur_losses, loss_sup, loss_unsup, isSUP=True)

            # lr = lr * (1 - epoch / args.epochs) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr
            # optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0
            tbar.set_description('T ({}) | Ls {:.2f} |'.format(epoch, loss_sup))

        writer.add_scalar('train/total_loss_epoch', (total_loss / (iters + 1)), epoch)
        metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 6)

        model.eval()
        tbar = tqdm(valloader)

        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model(x_l=img)
                pred = torch.argmax(pred, dim=1)

                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[1]
                classIOU = metric.evaluate()[0]

                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
                writer.add_scalar('test/mIOU_epoch', mIOU, epoch)

        mIOU *= 100.0

        print('mIOU: %.2f' % (mIOU))
        cIOU = classIOU.tolist()
        print('class1IOU: %.2f' % (cIOU[0] * 100.0))
        print('class2IOU: %.2f' % (cIOU[1] * 100.0))
        print('class3IOU: %.2f' % (cIOU[2] * 100.0))
        print('class4IOU: %.2f' % (cIOU[3] * 100.0))
        print('class5IOU: %.2f' % (cIOU[4] * 100.0))
        print('class6IOU: %.2f' % (cIOU[5] * 100.0))

        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(
                    os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))
        best_name = args.save_path + '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)
        #     best_model = deepcopy(model)
        #
        # if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
        #     checkpoints.append(deepcopy(model))
    #
    # if MODE == 'train':
    #     return best_model, checkpoints

    return best_name




def select_reliable(models, dataloader, args, istest):
    if istest:
        device, availble_gpus = _get_available_devices(1)
        models = torch.nn.DataParallel(models, device_ids=availble_gpus)
        models.to(device)

    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    # for i in range(len(models)):
    #     models[i].eval()

    models.eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        mIOUList = []
        metric = meanIOU(6)
        for img, mask, id in tbar:
            img = img.cuda()
            pred = models(img)
            pred = torch.argmax(pred, dim=1).cpu()
            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[1]
            mIOUList.append(mIOU)

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

            # reliability = sum(mIOU) / len(mIOU)
            reliability = sum(mIOUList) / len(mIOUList)
            id_to_reliability.append((id[0], reliability))

        mIOU *= 100.0
        print('mIOU: %.2f' % (mIOU))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')


def label(model, dataloader, args, istest):
    if istest:
        device, availble_gpus = _get_available_devices(1)
        model = torch.nn.DataParallel(model, device_ids=availble_gpus)
        model.to(device)

    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 6)
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            pred = model(img)
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[1]
            classIOU = metric.evaluate()[0]


            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)

            pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

        mIOU *= 100.0

        print('mIOU: %.2f' % (mIOU))

        cIOU = classIOU.tolist()

        print('class1IOU: %.2f' % (cIOU[0] * 100.0))
        print('class2IOU: %.2f' % (cIOU[1] * 100.0))
        print('class3IOU: %.2f' % (cIOU[2] * 100.0))
        print('class4IOU: %.2f' % (cIOU[3] * 100.0))
        print('class5IOU: %.2f' % (cIOU[4] * 100.0))
        print('class6IOU: %.2f' % (cIOU[5] * 100.0))



if __name__ == '__main__':
    args = parse_args()

    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 240, 'sodl': 160, 'ir': 320, 'whdld': 320}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004, 'sodl': 0.01, 'ir': 0.01, 'whdld': 0.01}[
                      args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 321, 'cityscapes': 721, 'sodl': 321, 'ir': 250, 'whdld': 250}[args.dataset]


    print(args)

    data = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/model/ssl/config.json'
    f = open(data, 'r', encoding='utf-8')
    config = json.load(f)
    main(args, config)
