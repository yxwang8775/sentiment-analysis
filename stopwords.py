import jieba
import pandas as pd


# 创建停用词list函数
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  # 分别读取停用词表里的每一个词，
    # 因为停用词表里的布局是一个词一行
    return stopwords  # 返回一个列表，里面的元素是一个个的停用词

# 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('stopwords/baidu_stopwords.txt')  # 这里加载停用词的路径,同时调用上面的stopwordslist()函数
    outstr = ''  # 设置一个空的字符串，用于储存结巴分词后的句子
    for word in sentence_seged:  # 遍历分词后的每一个单词
        if word not in stopwords:  # 如果这个单词不在停用表里面
            if word != '\t':  # 且这个单词不是制表符
                outstr += word  # 就将这个词添加到结果中
                outstr += " "  # 但是这里为什么又要添加双引号中间带空格？
                # 测试了一下，原来是为了让结巴分词后的词以空格间隔分割开
    return outstr

inputss = pd.read_excel('baiduzd_questions.xlsx')

outputs = open('output.txt', 'w')

for line in inputss.itertuples():  # 使用a.itertuples()遍历DataFrame的每一行
    linE = getattr(line, 'title')  # 获得每一行
    line_seg = seg_sentence(linE)  # 对每一行调用上面的seg_sentence（）函数，返回值是字符串
    outputs.write(line_seg + '\n')  # 换行输入

outputs.close()
