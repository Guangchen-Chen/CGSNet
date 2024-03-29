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





    # <====================== 第一阶段一致性正则化训练 ======================>
    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (8))

    global MODE

    #设置数据增强模式 train:弱数据增强
    MODE = 'train'
    unlabeledtrain_id_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/dataset/splits/sodl/1_16/split_0/unlabeled_train.txt'

    trainset_unsup = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, unlabeledtrain_id_path)
    trainloader_unsup = DataLoader(trainset_unsup, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=6, drop_last=True)

    num_classes = 6


    # SUPERVISED LOSS

    sup_loss = CE_loss


    # MODEL
    rampup_ends = int(config['ramp_up'] * args.epochs)
    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=len(trainloader_unsup),
                                        rampup_ends=rampup_ends)

    ctmodel = CCT(num_classes=num_classes, conf=config['model'],
    						sup_loss=sup_loss, cons_w_unsup=cons_w_unsup,
    						weakly_loss_w=config['weakly_loss_w'], use_weak_lables=config['use_weak_lables'],
                            ignore_index=255)

    #--------------------
    model, optimizer = init_basic_elems(args)
    #------------------

    #手动模型载入
    best_name = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/exprements/ir/id=71/deeplabv3plus_resnet50_57.01.pth'
    best_model_path = best_name
    # best_model = ctmodel
    # best_model.load_state_dict(torch.load(best_model_path))

    best_model = model
    best_model.module.load_state_dict(torch.load(best_model_path))

    # valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    # valloader = DataLoader(valset, batch_size=1 if args.dataset == 'cityscapes' else 6,
    #                        shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    cur_unlabeled_id_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/dataset/splits/whdld/val.txt'
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args, istest=True)
    # label(best_model, dataloader, args, istest=True)


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

            pred.save('%s/%s' % ('/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/predictions', os.path.basename(id[0].split(' ')[1])))

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
        args.epochs = {'pascal': 80, 'cityscapes': 240, 'sodl': 320, 'ir': 80, 'whdld': 80}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004, 'sodl': 0.01, 'ir': 0.01, 'whdld': 0.01}[
                      args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 321, 'cityscapes': 721, 'sodl': 320, 'ir': 250, 'whdld': 250}[args.dataset]


    print(args)

    data = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/model/ssl/config.json'
    f = open(data, 'r', encoding='utf-8')
    config = json.load(f)
    main(args, config)
