#!/usr/bin/env python
# a.txt 的文件需要复制多行数据，每行读取每一行 a.txt 数据 ，依次写入 b.txt
num = 0
f = open("/home/bj/projects/Semi-supervised/ST-PlusPlus-master/dataset/splits/sodl/1_8/split_0/unlabeled.txt","r",encoding='utf-8')
for line in f.readlines():
        with open("/home/bj/projects/Semi-supervised/ST-PlusPlus-master/dataset/splits/sodl/1_8/split_0/unlabeled_train.txt", "a") as mon:
            if num < 160:
                mon.write(line)
                num += 1

f.close()

