from sklearn.model_selection import train_test_split
import os

def readTxt():
    data = []
    with open("/home/bj/projects/Semi-supervised/ST-PlusPlus-master/dataset/splits/sodl/1_4/split_0/unlabeled.txt","r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            line = line.split()
            data.append(line[-2]+ "\n")
    #print(data)
    return data



if __name__ == '__main__':
    data = readTxt()
    full_path = '/home/bj/projects/Semi-supervised/ST-PlusPlus-master/dataset/splits/sodl/1_4/split_0/unlabeled_cps.txt'
    f = open(full_path, "w")
    f.writelines(data)
    f.close()

