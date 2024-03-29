from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map, compute_sdf
from Tools import losses

import time
import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from tensorboardX import SummaryWriter



MODE = None

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 1.0 * sigmoid_rampup(epoch, 40.0)

def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes', 'sodl'], default='pascal')
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


def main(args):
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

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (6))

    global MODE
    MODE = 'train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=12, drop_last=True)

    model, optimizer = init_basic_elems(args)
    print('\nParams: %.1fM' % count_params(model))

    best_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args, 'SupOnly train')

    """
        ST framework without selective re-training
    """

    ## <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')

    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    select_reliable(checkpoints, dataloader, args)

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args)

    # <======================== Re-training on labeled and unlabeled images ========================>
    print('\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images')

    MODE = 'semi_train'


    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)

    train(model, trainloader, valloader, criterion, optimizer, args, 'semi_train')

    return


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


def train(model, trainloader, valloader, criterion, optimizer, args, flag):

    log = "./logs/"
    if not os.path.exists(log+flag):
        os.makedirs(log+flag)
    logdir = log+flag
    writer = SummaryWriter(logdir)

    iters = 0
    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0

    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()

    labeled_bs = 8

    global MODE

    if MODE == 'train':
        checkpoints = []

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader)

        for i, (img, mask) in enumerate(tbar):
            time2 = time.time()
            img, mask = img.cuda(), mask.cuda()

            outputs, outputs_tanh= model(img)
            outputs_soft = torch.sigmoid(outputs)

            # calculate the loss
            with torch.no_grad():
                gt_dis = compute_sdf(mask[:].cpu().numpy(), outputs[:labeled_bs, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()

            loss_sdf = mse_loss(outputs_tanh[:labeled_bs, 0, ...], gt_dis)
            loss_seg = ce_loss(outputs[:labeled_bs, 0, ...], mask[:labeled_bs].float())

            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 0], mask[:labeled_bs] == 1)
            dis_to_mask = torch.sigmoid(-1500 * outputs_tanh)

            consistency_loss = torch.mean((dis_to_mask - outputs_soft) ** 2)
            supervised_loss = loss_seg_dice + 0.3 * loss_sdf
            consistency_weight = get_current_consistency_weight(i // 150)

            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))
            writer.add_scalar('train/total_loss_iter', (total_loss / (i + 1)), i)

        writer.add_scalar('train/total_loss_epoch', (total_loss / iters), epoch)
        metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 6)
        # #查看权重直方图
        # for name, param in model.named_parameters():
        #     writer.add_histogram(tag=name+"_grad", values=param.grad, global_step=epoch)
        #     writer.add_histogram(tag=name+'_data', values=param.data, global_step=epoch)

        model.eval()
        tbar = tqdm(valloader)

        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model(img)[0]
                pred = torch.argmax(pred, dim=1)
                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[1]
                classIOU = metric.evaluate()[0]
                classPA = metric.evaluate()[2]
                mPA = metric.evaluate()[3]
                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
                writer.add_scalar('test/mIOU_epoch', mIOU, epoch)

        mIOU *= 100.0
        mPA *= 10.0
        print('mIOU: %.2f' % (mIOU))
        print('mPA: %.2f' % (mPA))
        cIOU = classIOU.tolist()
        cPA = classPA.tolist()
        print('class1IOU: %.2f' % (cIOU[0] * 100.0))
        print('class2IOU: %.2f' % (cIOU[1] * 100.0))
        print('class3IOU: %.2f' % (cIOU[2] * 100.0))
        print('class4IOU: %.2f' % (cIOU[3] * 100.0))
        print('class5IOU: %.2f' % (cIOU[4] * 100.0))
        print('class6IOU: %.2f' % (cIOU[5] * 100.0))
        print('class1PA: %.2f' % (cPA[0] * 10.0))
        print('class2PA: %.2f' % (cPA[1] * 10.0))
        print('class3PA: %.2f' % (cPA[2] * 10.0))
        print('class4PA: %.2f' % (cPA[3] * 10.0))
        print('class5PA: %.2f' % (cPA[4] * 10.0))
        print('class6PA: %.2f' % (cPA[5] * 10.0))

        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

            best_model = deepcopy(model)

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints

    return best_model


def select_reliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()

            preds = []
            for model in models:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 6)
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')


def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 6)
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            pred = model(img, True)
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[1]
            classIOU = metric.evaluate()[0]
            classPA = metric.evaluate()[2]
            mPA = metric.evaluate()[3]

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)

            pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
        mIOU *= 100.0
        mPA *= 10.0
        print('mIOU: %.2f' % (mIOU))
        print('mPA: %.2f' % (mPA))
        cIOU = classIOU.tolist()
        cPA = classPA.tolist()
        print('class1IOU: %.2f' % (cIOU[0] * 100.0))
        print('class2IOU: %.2f' % (cIOU[1] * 100.0))
        print('class3IOU: %.2f' % (cIOU[2] * 100.0))
        print('class4IOU: %.2f' % (cIOU[3] * 100.0))
        print('class5IOU: %.2f' % (cIOU[4] * 100.0))
        print('class6IOU: %.2f' % (cIOU[5] * 100.0))
        print('class1PA: %.2f' % (cPA[0] * 10.0))
        print('class2PA: %.2f' % (cPA[1] * 10.0))
        print('class3PA: %.2f' % (cPA[2] * 10.0))
        print('class4PA: %.2f' % (cPA[3] * 10.0))
        print('class5PA: %.2f' % (cPA[4] * 10.0))
        print('class6PA: %.2f' % (cPA[5] * 10.0))


if __name__ == '__main__':
    args = parse_args()

    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 240, 'sodl': 160}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004, 'sodl': 0.001}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 321, 'cityscapes': 721, 'sodl': 321}[args.dataset]

    print()
    print(args)

    main(args)
