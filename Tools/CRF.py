import numpy as np
import pydensecrf.densecrf as dcrf
import os
import cv2
from PIL import Image
from Tools.FillHoel  import area_connection
from collections import namedtuple
import time, sys
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

"""
original_image_path  原始图像路径
predicted_image_path  之前用自己的模型预测的图像路径
CRF_image_path  即将进行CRF后处理得到的结果图像保存路径
"""

# 条件随机场函数
def CRFs(img, predicted_image_path):

    # 将predicted_image的RGB颜色转换为uint32颜色 0xbbggrr
    anno_rgb = predicted_image_path.astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # 将uint32颜色转换为1,2,...
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # 如果你的predicted_image里的黑色（0值）不是待分类类别，表示不确定区域，即将分为其他类别
    # 那么就取消注释以下代码
    # HAS_UNK = 0 in colors
    # if HAS_UNK:
    # colors = colors[1:]

    # 创建从predicted_image到32位整数颜色的映射。
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # 计算predicted_image中的类数。
    n_labels = len(set(labels.flat))
    # n_labels = len(set(labels.flat)) - int(HAS_UNK) ##如果有不确定区域，用这一行代码替换上一行

    ###########################
    ###     设置CRF模型     ###
    ###########################
    use_2d = False
    # use_2d = True

    if use_2d:
        # 使用densecrf2d类
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 增加了与颜色无关的术语，功能只是位置而已
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 增加了颜色相关术语，即特征是(x,y,r,g,b)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        # 使用densecrf类
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # 得到一元势（负对数概率）
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=None)
        # U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)## 如果有不确定区域，用这一行代码替换上一行
        d.setUnaryEnergy(U)

        # 这将创建与颜色无关的功能，然后将它们添加到CRF中
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # 这将创建与颜色相关的功能，然后将它们添加到CRF中
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ###         做推理和计算         ###
    ####################################

    # 进行5次推理
    Q = d.inference(5)

    # 找出每个像素最可能的类
    MAP = np.argmax(Q, axis=0)

    # 将predicted_image转换回相应的颜色并保存图像
    MAP = colorize[MAP, :]
    out = MAP.reshape(img.shape)

    # cv2.imwrite(CRF_image_path + filename[:-4] + '.png', out)
    return out

# 抑制噪声
def Point_Noise_remove(imgPath, SavePath, filenames):
    filename = imgPath
    img = cv2.imread(filename, 0)
    print(np.shape(img))
    kernel = np.ones((7, 7), np.uint8)
    dilate = cv2.dilate(img, kernel, iterations=1)
    # cv2.imwrite('./lishuwang_dilate.jpg', dilate)
    # erosion = cv2.erode(img,kernel,iterations = 1)
    # cv2.imwrite('lishuwang_erosion.jpg',erosion)
    canny1 = cv2.Canny(dilate, 10, 100)
    # cv2.imwrite('./lishuwang_canny.jpg', canny1)
    # kernel2 = np.ones((2,1),np.uint8)
    # erosion = cv2.erode(canny,kernel2,iterations = 1)
    # cv2.imwrite('lishuwang_erosion.jpg',erosion)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(canny1)
    print(centroids)
    print("stats", stats)
    i = 0

    for istat in stats:
        if istat[4] < 100:
            # print(i)
            print(istat[0:2])
            if istat[3] > istat[4]:
                r = istat[3]
            else:
                r = istat[4]
            cv2.rectangle(img, tuple(istat[0:2]), tuple(istat[0:2] + istat[2:4]), 0, thickness=-1)  # 26
        i = i + 1

    cv2.imwrite(SavePath + filenames[:-4] + 'pn.png', img)

def arrays_jd(arrays, cond_nums):
    r = arrays[0] == cond_nums[0]
    g = arrays[1] == cond_nums[1]
    b = arrays[2] == cond_nums[2]
    return r & g & b

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

# 彩色转换数字标签函数
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
    gt_list.sort()
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

def CRF(original_image_path, predicted_image_path, fill_out):
    #遍历该目录下的所有图片文件
    print('CRFs+fill_hole --- start: ')
    Cls = namedtuple('cls', ['name', 'id', 'color'])

    Clss = [
        Cls('背景', 0, (0, 0, 0)),
        Cls('built-up', 1, (128, 0, 0)),
        Cls('farmland', 2, (0, 128, 0)),
        Cls('forest', 3, (128, 128, 0)),
        Cls('water', 4, (0, 0, 128)),
        Cls('wetlands', 5, (128, 0, 128))
    ]

    color_dict = nt_dic(Clss)

    colormap = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                [128, 0, 0], [128, 0, 128]]
    i = 0
    for filename in os.listdir(predicted_image_path):
        print(i)
        pre_img = cv2.imread(predicted_image_path+'/'+filename)
        oriname = filename[:-4] + '.jpg'
        ori_img = cv2.imread(original_image_path+oriname)
        # 条件随机场
        crf_img = CRFs(ori_img, pre_img)
        # 填补空洞
        im_out = area_connection(crf_img, colormap)
        # 保存
        cv2.imwrite(fill_out + filename[:-4] + '.png', im_out.astype(np.uint8))
        i += 1

    print('Trans_label --- start: ')
    # 彩色转换数字标签
    color_gray(color_dict, color_path=fill_out, gray_path=fill_out)



    # print('FillHole—start: ')
    # for filename in os.listdir(CRF_image_path):
    #     in_path = CRF_image_path+'/'+filename
    #     print(filename)
    #     pre_img = cv2.imread(in_path).astype(np.float32)
    #     colormap = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
    #                 [128, 0, 0], [128, 0, 128]]
    #     im_out = area_connection(pre_img, colormap)
    #
    #     cv2.imwrite(fill_out + filename[:-4] + '.png', im_out.astype(np.uint8))



if __name__ == '__main__':
    original_image_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/Tools/test_img/'
    predicted_image_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/Tools/hot_png'
    CRF_image_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/Tools/crfs_out/'
    fill_out = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/Tools/Fill_out/'


    CRF(original_image_path, predicted_image_path, CRF_image_path)
