#-*- coding : utf-8 -*-
# coding: utf-8
import csv

import re

import argparse

import pandas as pd


def get_data(data_path):
    csvFileObj = open(data_path, encoding="utf-8")
    lines = csv.reader(csvFileObj)
    text_list = []
    label_list = []

    for line in lines:
        if not (line[1] in {"0","1","2"}):
            continue;
        label = int(line[1])
        text = line[0]
        text_list.append(text)
        label_list.append(label)
    return text_list,label_list

def save_data(text_list, label_list, path):
    data=zip(text_list,label_list)
    output = pd.DataFrame(data=data)
    output.to_csv('save_path', header=None, index=None,encoding="utf-8")




#删去无用文本段
def del_text(text_list):
        pattern_list=[]
        pattern_list.append(re.compile("//@[^:]*:(转发微博)?|回复@[^:]*:"))
        pattern_list.append(re.compile("#[^#]*#|【[^】]*】"))
        pattern_list.append(re.compile("\?展开全文c|O网页链接\?*|查看图片|转发微博|\ue627"))

        for text in text_list:
            for pattern in pattern_list:
                text=pattern.sub("",text)

        return text_list



def main():
    data_path=args.f
    output_path=args.o
    text_list, label_list=get_data(data_path)
    print(f'len1={len(text_list)}')
    text_list=del_text(text_list)
    save_data(text_list, label_list, output_path)
    print(f'len2={len(text_list)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="file path")
    parser.add_argument("-o", help="output file path")
    args = parser.parse_args()
    main()

