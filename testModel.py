from PIL import Image
from tqdm import tqdm

import os, sys, time, cv2
import numpy as np
from collections import namedtuple
import png # https://pypng.readthedocs.io/en/latest/

#
# Cls = namedtuple('cls', ['name', 'id', 'color'])
# Clss = [
#     Cls('bg', (0, 0, 0), (0, 0, 0)),
#     Cls('cls1', (38, 38, 38), (128, 0, 0)),
#     Cls('cls2', (113, 113, 113), (128, 128, 0)),
#     Cls('cls3', (75, 75, 75), (0, 128, 0)),
#     Cls('cls4', (15, 15, 15), (0, 0, 128)),
#     Cls('cls5', (53, 53, 53), (128, 0, 128))
# ]
#
#
# # region 灰度转彩色
# def get_color8bit(grays_path, colors_path):
#     '''
#     灰度图转8bit彩色图
#     :param grays_path:  灰度图文件路径
#     :param colors_path: 彩色图文件路径
#     :return:
#     '''
#     if not os.path.exists(colors_path):
#         os.makedirs(colors_path)
#     file_names = os.listdir(grays_path)
#     bin_colormap = get_putpalette(Clss)
#     with tqdm(file_names) as pbar:
#         for file_name in pbar:
#             gray_path = os.path.join(grays_path, file_name)
#             color_path = os.path.join(colors_path, file_name.replace('.tif', '.png'))
#             gt = Image.open(gray_path)
#             gt.putpalette(bin_colormap)
#             gt.save(color_path)
#             pbar.set_description('get color')
#
#
# def get_putpalette(Clss, color_other=[0, 0, 0]):
#     '''
#     灰度图转8bit彩色图
#     :param Clss:颜色映射表
#     :param color_other:其余颜色设置
#     :return:
#     '''
#     putpalette = []
#     for cls in Clss:
#         putpalette += list(cls.color)
#     putpalette += color_other * (255 - len(Clss))
#     return putpalette
#


# 类别信息
Cls = namedtuple('cls', ['name', 'id', 'color'])
Clss1 = [
    Cls('bg', 0, (0, 0, 0)),
    Cls('cls1', 38, (128, 0, 0)),
    Cls('cls2', 113, (128, 128, 0)),
    Cls('cls3', 75, (0, 128, 0)),
    Cls('cls4', 15, (0, 0, 128)),
    Cls('cls5', 53, (128, 0, 128))
]

Clss = [
    Cls('背景', 0, (0, 0, 0)),
    Cls('built-up', 1, (128, 0, 0)),
    Cls('farmland', 2, (0, 128, 0)),
    Cls('forest', 3, (128, 128, 0)),
    Cls('water', 4, (0, 0, 128)),
    Cls('wetlands', 5, (128, 0, 128))
]


def gray_color(color_dict, gray_path, color_path):
    '''
    swift gray image to color, by color mapping relationship
    :param color_dict:color mapping relationship, dict format
    :param gray_path:gray imgs path
    :param color_path:color imgs path
    :return:
    '''
    pass
    t1 = time.time()
    gt_list = os.listdir(gray_path)
    for index, gt_name in enumerate(gt_list):
        gt_gray_path = os.path.join(gray_path, gt_name)
        gt_color_path = os.path.join(color_path, gt_name)
        gt_gray = cv2.imread(gt_gray_path, cv2.IMREAD_GRAYSCALE)
        assert len(gt_gray.shape) == 2  # make sure gt_gray is 1band

        # # region method 1: swift by pix, slow
        # gt_color = np.zeros((gt_gray.shape[0],gt_gray.shape[1],3),np.uint8)
        # for i in range(gt_gray.shape[0]):
        #     for j in range(gt_gray.shape[1]):
        #         gt_color[i][j] = color_dict[gt_gray[i][j]]      # gray to color
        # # endregion

        # region method 2: swift by array
        # gt_color = np.array(np.vectorize(color_dict.get)(gt_gray),np.uint8).transpose(1,2,0)
        # endregion

        # region method 3: swift by matrix, fast
        gt_color = matrix_mapping(color_dict, gt_gray)
        # endregion
        gt_color = cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(gt_color_path, gt_color)
        process_show(index + 1, len(gt_list))
    print(time.time() - t1)


def color_gray(color_dict, color_path, gray_path):
    '''
    swift color image to gray, by color mapping relationship
    :param color_dict:color mapping relationship, dict format
    :param gray_path:gray imgs path
    :param color_path:color imgs path
    :return:
    '''
    gray_dict = {}
    for k, v in color_dict.items():
        gray_dict[v] = k
    t1 = time.time()
    gt_list = os.listdir(color_path)
    for index, gt_name in enumerate(gt_list):
        gt_gray_path = os.path.join(gray_path, gt_name)
        gt_color_path = os.path.join(color_path, gt_name)
        color_array = cv2.imread(gt_color_path, cv2.IMREAD_COLOR)
        assert len(color_array.shape) == 3

        gt_gray = np.zeros((color_array.shape[0], color_array.shape[1]), np.uint8)
        b, g, r = cv2.split(color_array)
        color_array = np.array([r, g, b])
        for cls_color, cls_index in gray_dict.items():
            cls_pos = arrays_jd(color_array, cls_color)
            gt_gray[cls_pos] = cls_index

        cv2.imwrite(gt_gray_path, gt_gray)
        process_show(index + 1, len(gt_list))
    print(time.time() - t1)


def arrays_jd(arrays, cond_nums):
    r = arrays[0] == cond_nums[0]
    g = arrays[1] == cond_nums[1]
    b = arrays[2] == cond_nums[2]
    return r & g & b


def matrix_mapping(color_dict, gt):
    colorize = np.zeros([255, 3], 'uint8')
    for cls, color in color_dict.items():
        colorize[cls, :] = list(color)
    ims = colorize[gt, :]
    ims = ims.reshape([gt.shape[0], gt.shape[1], 3])
    return ims


def nt_dic(nt):
    '''
    swift nametuple to color dict
    :param nt: nametuple
    :return:
    '''
    pass
    color_dict = {}
    for cls in nt:
        color_dict[cls.id] = cls.color
    return color_dict


def process_show(num, nums, pre_fix='', suf_fix=''):
    '''
    auxiliary function, print work progress
    :param num:
    :param nums:
    :param pre_fix:
    :param suf_fix:
    :return:
    '''
    rate = num / nums
    ratenum = round(rate, 3) * 100
    bar = '\r%s %g/%g [%s%s]%.1f%% %s' % \
          (pre_fix, num, nums, '#' * (int(ratenum) // 5), '_' * (20 - (int(ratenum) // 5)), ratenum, suf_fix)
    sys.stdout.write(bar)
    sys.stdout.flush()

p = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0,
128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128,
192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128,
192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64,
0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128,
0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192,
128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0,
192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0,
64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192,
64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0,
160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128,
128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128,
128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160,
64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0,
96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128,
64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224,
0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192,
32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192,
192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224,
64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0, 0, 32,
128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160,
0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0,
128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64,
224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128,
32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64,
32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192,
160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224,
192, 128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96,
192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128,
160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96,
32, 128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224,
0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0,
224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64,
32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64,
224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192,
32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192,
160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192,
96, 224, 192, 224, 224, 192]

def convertPNG(gray_path, color_path):

    gt_list = os.listdir(gray_path)
    for index, gt_name in enumerate(gt_list):
        gt_gray_path = os.path.join(gray_path, gt_name)
        gt_color_path = os.path.join(color_path, gt_name)
        # 读取灰度图


        empire = Image.open(gt_gray_path)
        mask =  np.array(Image.open(gt_gray_path))
        h, w, c = mask.shape
        print(empire.mode)  # RGB
        pre = empire.convert('P')

        # image = Image.new('P',(h, w))

        # image.paste(mask, (0, 0, h, w))
        pre.putpalette(p)
        print(pre.mode)
        pre.save(gt_color_path)

def newP(in_path, out_path):
    '''
    128,0,0  ->  153,0,0  102,0,0
    0,128,0  ->  0,153,0  102,0,0
    '''
    imgs_list = os.listdir(in_path)
    imgs_list.sort()
    for index, name in enumerate(imgs_list):
        in_path = os.path.join(in_path, name)
        out_path = os.path.join(out_path, name)

        img = np.array(Image.open(in_path))
        h, w = img.shape

        for row in range(h):
            for col in range(w):
                b = img[row, col]









if __name__ == '__main__':
    path1 = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/pseudo_masks/sodl/1_4/fill/'
    path2 = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/pseudo_masks/sodl/1_4/split_0'
    savedir = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/pseudo_masks/sodl/1_4/out'


    color_dict = nt_dic(Clss)
    color_gray(color_dict, color_path=path1, gray_path=savedir)
    # files = os.listdir(path1)
    # files.sort()
    # i = 0
    # for filename in files:
    #     # if i > 5 :
    #     #     break
    #     print(filename)
    #     img = cv2.imread(path1 + '/' + filename)
    #     cv2.imshow("image1", img)
    #     imgs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     cv2.imshow("image2", imgs)
    #     # cv2.waitKey(0)
    #     cv2.imwrite(savedir + filename[:-4] + '.png', imgs)
    #     # i += 1

    grays_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/pseudo_masks/sodl/1_4/out/'
    colors_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/outdir/pseudo_masks/sodl/1_4/mask'


    # gray_color(color_dict, gray_path=grays_path, color_path=colors_path)
    # convertPNG(path1, savedir)

    # newP(grays_path, colors_path)

