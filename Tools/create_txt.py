from sklearn.model_selection import train_test_split
import os
import random
import linecache


def split_val_train(imagedir, maskdir, outdir):
    seed = 1
    images = []
    masks = []
    nums = 0
    path1_list = os.listdir(imagedir)
    path2_list = os.listdir(maskdir)
    path1_list.sort()
    path2_list.sort()

    for file in path1_list:
        filename = 'JPEGImages/' + file
        images.append(filename)
        nums += 1

    for file in path2_list:
        filename = 'SegmentationClass/' + file
        masks.append(filename)

    val, train = train_test_split(images, train_size=0.2, random_state=seed)
    valmask, trainmask = train_test_split(masks, train_size=0.2, random_state=seed)

    with open(outdir + "val.txt", 'w') as f:
        val_num = len(val)
        for i in range(0, val_num):
            f.write(val[i] + ' ' + valmask[i] + '\n')

    with open(outdir + "train.txt", 'w') as f:
        train_num = len(train)
        for i in range(0, train_num):
            f.write(train[i] + ' ' + trainmask[i] + '\n')


def split_labeled_unlebeled(train_path, split_path, split=1.0):

    labeled_path = split_path + 'labeled.txt'
    unlabeled_path = split_path + 'unlabeled.txt'

    with open(train_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # 获取所有行
        list = []
        for line in lines:  # 第i行
            # 找到第一个空格
            list.append(line)

    seed = 0

    labeled, unlabeled = train_test_split(list, train_size=split, random_state=seed)

    with open(labeled_path, 'w') as f:
        labeled_num = len(labeled)
        for i in range(0, labeled_num):
            f.write(labeled[i])

    with open(unlabeled_path, 'w') as f:
        unlabeled_num = len(unlabeled)
        for i in range(0, unlabeled_num):
            f.write(unlabeled[i])

    f.close()

def unlabel_split(unlabeled_path, semi_path, split=1.0):

    semi_file = semi_path + 'unlabeled_train.txt'

    num = 0

    f = open(unlabeled_path, "r", encoding='utf-8')

    nums = int(1280*split)
    for line in f.readlines():
        with open( semi_file,"a") as mon:
            if num < nums:
                mon.write(line)
                num += 1

    f.close()


if __name__ == '__main__':

    imagedir = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/WHDLD/JPEGImages/'
    maskdir = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/WHDLD/SegmentationClass/'
    outdir = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/dataset/splits/whdld/'

    # 划分训练集与测试集
    # split_val_train(imagedir, maskdir, outdir)

    train_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/dataset/splits/whdld/train.txt'
    out_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/dataset/splits/whdld/1_4/split_0/'

    # 划分有标签与无标签数据, split是比例
    # split_labeled_unlebeled(train_path, out_path, split=1/4)

    unlabeled_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/dataset/splits/whdld/1_16/split_0/unlabeled.txt'
    semi_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/dataset/splits/whdld/1_16/split_0/'

    # 随机抽取相同数量的无标签数据
    unlabel_split(unlabeled_path, semi_path, split=1/10)

