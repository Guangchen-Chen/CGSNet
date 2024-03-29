import cv2
import numpy as np
import os
import skimage
import tqdm
from keras.utils.np_utils import to_categorical
from skimage import morphology
import scipy.ndimage
import scipy as sp


def colormap_voc():
    """
    create a colormap
    """
    colormap = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                [128, 0, 0], [128, 0, 128]]

    classes = ['background', 'built-up', 'farmland', 'forest',
               'water', 'wetlands']

    return colormap, classes


def label_to_onehot(label, colormap):
    """
    Converts a segmentation label (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in colormap:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map

def onehot_to_label(semantic_map, colormap):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(semantic_map, axis=-1)
    colour_codes = np.array(colormap)
    label = np.uint8(colour_codes[x.astype(np.uint8)])
    return label

def fill(test_array,h_max=255):
    input_array = np.copy(test_array)
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array

def fill_contours(img, seedPoint):

    pred = np.array(img, np.uint8)

    im_floodfill = pred.copy()

    # Mask 用于 floodFill，官方要求长宽+2
    h, w = pred.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # floodFill函数中的seedPoint对应像素必须是背景
    # isbreak = False
    # for i in range(im_floodfill.shape[0]):
    #     for j in range(im_floodfill.shape[1]):
    #         if (im_floodfill[i][j] == 0):
    #             seedPoint = (i, j)
    #             isbreak = True
    #             break
    #     if (isbreak):
    #         break

    # 得到im_floodfill 255填充非孔洞值
    cv2.floodFill(im_floodfill, mask, seedPoint, 255)

    # 得到im_floodfill的逆im_floodfill_inv
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
    im_out = pred | im_floodfill_inv

    return im_out


def fill_hole(img, colormap):
    gray = np.array(img, np.uint8)
    img = np.array(img, np.uint8)
    ret, binary = cv2.threshold(gray, 0, 10, cv2.THRESH_BINARY)  # 图像二值化

    # img_contour, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找物体轮廓

    img_contours = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        # print("轮廓 %d 的面积是:%d" % (i, area))
        if area > 100 :
            img_contours.append(contours[i])


    contours = cv2.drawContours(img, img_contours, -1, colormap, cv2.FILLED)  # 绘制所有轮廓
    # cv2.imshow("contour", contours)
    # cv2.waitKey(0)
    return contours

def area_connection(result, colormap):
    """
    result:预测影像
    area_threshold：最小连通尺寸，小于该尺寸的都删掉
    """
    # result = to_categorical(result, num_classes=n_class, dtype='uint8')  # 转为one-hot
    result = label_to_onehot(result, colormap).astype(np.float32)
    # map = np.array(result, np.uint8)
    # seedPoint = (0, 0)
    # isbreak = False
    # for i in range(map.shape[0]):
    #     for j in range(map.shape[1]):
    #         isSeed = True
    #         for k in range(map.shape[2]):
    #             if k != 0:
    #                 if (map[i][j][k] != 0):
    #                     isSeed = False
    #         if isSeed:
    #             seedPoint = (i, j)
    #             isbreak = True
    #             break
    #     if isbreak:
    #         break

    for i in range(len(colormap)):
        # image = (result[:, :, i] == 1)
        if i != 0:
            # 去除孔洞
            result[:, :, i] = morphology.remove_small_holes(result[:, :, i] == 1, area_threshold=64,
                                                            connectivity=1, in_place=True)
            # 去除小物体
            result[:, :, i] = morphology.remove_small_objects(result[:, :, i] == 1, min_size=16,
                                                              connectivity=1, in_place=True)

        # 空洞修补程序
        image = result[:, :, i]
        mask = fill_hole(image, i)
        result[:, :, i] = np.array(mask, np.float32)

    # 获取最终label
    # result = np.argmax(result, axis=2).astype(np.uint8)

    out = onehot_to_label(result, colormap)

    return out

def main(imgPath, SavePath):
    #遍历该目录下的所有图片文件
    files = os.listdir(imgPath)
    files.sort()
    for filename in files:
        print(filename)
        pre_img = cv2.imread(imgPath+'/'+filename).astype(np.float32)
        colormap = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                    [128, 0, 0], [128, 0, 128]]
        im_out = area_connection(pre_img, colormap)

        cv2.imwrite(SavePath + filename[:-4] + '.png', im_out.astype(np.uint8))




if __name__ == '__main__':
    imgPath = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/Tools/crfs_out'
    SavePath = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/Tools/hole_out/'
    main(imgPath, SavePath)