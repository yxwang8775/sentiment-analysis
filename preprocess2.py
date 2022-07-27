#-*- coding : utf-8 -*-
# coding: utf-8

import random
#打乱顺序 并按照8/1/1划分并行句子数据集
src=open("data/data_all3.csv", "r", encoding='utf-8').readlines()

datasize=len(src)

index_list=[i for i in range(datasize)]
random.shuffle(index_list)
src_train=[]
src_dev=[]
src_test=[]


for i in range(datasize):
    if(i%10<8):
        src_train.append(src[i])

    elif(i%10<9):
        src_dev.append(src[i])

    else:
        src_test.append(src[i])

file = open('data/data_train.csv', 'w',encoding='utf-8')
for line in src_train:
    file.writelines(line)
file.close()

file = open('data/data_dev.csv', 'w',encoding='utf-8')
for line in src_dev:
    file.writelines(line)
file.close()

file = open('data/data_test.csv', 'w',encoding='utf-8')
for line in src_test:
    file.writelines(line)
file.close()
