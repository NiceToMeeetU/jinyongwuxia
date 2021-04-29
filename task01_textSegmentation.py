# -*- coding: utf-8 -*-
# @Time    : 20/03/14 15:53
# @Author  : Wang Yu
# @Project : jinyongwuxia
# @File    : task01_textSegmentation.py
# @Software: PyCharm

"""
用于处理原始文档和初步提取的命名实体，任务完成后得到28篇分好词的文档及主要人物表
"""

import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import MYTOOL

entity_path = "data/entity/entity_person_"
raw_path = "data/raw_金庸原著/"
seg_path = "data/seg_金庸原著/"
part_path = "data/participle_金庸原著/"
SUFFIX = ["（新修版）.txt", "（三联版）.txt"]

BOOKS = ['书剑恩仇录', '侠客行', '倚天屠龙记', '天龙八部', '射雕英雄传', '白马啸西风', '碧血剑',
         '神雕侠侣', '笑傲江湖', '连城诀', '雪山飞狐', '飞狐外传', '鸳鸯刀', '鹿鼎记']
ROLES = "data/14本书主要人物出现次数（清理）.txt"


# 合并统计两版书人物分别出现次数：
def counts(book):
    names1 = [i.strip().split('\t')[0] for i in MYTOOL.read_txt_file(entity_path + book + SUFFIX[0], 0, 0) if
              int(i.strip().split('\t')[1]) >= 100]
    names2 = [i.strip().split('\t')[0] for i in MYTOOL.read_txt_file(entity_path + book + SUFFIX[1], 0, 0) if
              int(i.strip().split('\t')[1]) >= 100]
    names = sorted(list(set(names1 + names2)))
    texts1 = '\n'.join(
        [i.strip() for i in MYTOOL.read_txt_file(raw_path + book + SUFFIX[0], 0, 0) if len(i.strip()) > 0])
    texts2 = '\n'.join(
        [i.strip() for i in MYTOOL.read_txt_file(raw_path + book + SUFFIX[1], 0, 0) if len(i.strip()) > 0])

    D = [[i, texts1.count(i) + texts2.count(i), texts1.count(i), texts2.count(i)] for i in names]
    # 按总的出现次数排序
    D = sorted(D, key=(lambda x: x[1]), reverse=True)
    return D


# 汇总14本书所有人物出现频次，集中手工清洗
def summary():
    with open("data/14本书主要人物出现次数（原始）.txt", 'a+', encoding='utf-8') as f:
        for i in BOOKS:
            print("$$$", file=f)
            print(i, file=f)
            d = counts(i)
            for j in d:
                j = [j[0], str(j[1]), str(j[2]), str(j[3])]
                print(' '.join(j), file=f)


# 将清理好的人物统计改成自定义词典形式保存
def build_dict():
    with open("data/dict_p.txt", 'a+', encoding='utf-8') as f:
        raw = MYTOOL.read_txt_file("data/14本书主要人物出现次数（清理）.txt", 0, 0)
        dict_p = [i.split()[0] for i in raw if len(i.split()) > 3]
        for j in dict_p:
            print(j, file=f)


# 按照章节分割每个文档
def seg_text(book):
    for i in [0, 1]:
        texts = MYTOOL.read_txt_file(seg_path + book + SUFFIX[i], 0, 0)
        texts = '\n'.join([line.strip() for line in texts if len(line.strip()) > 0])
        segs = texts.split('$$$')
        print(book, SUFFIX[i], len(segs))


# 批量分割金庸原文
def seg_texts():
    for book in BOOKS:
        seg_text(book)


# 对分割好的文档进一步按照自定义词典分词处理
def participle_book(book):
    jieba.load_userdict("dict_p.txt")
    for i in [0, 1]:
        texts = MYTOOL.read_txt_file(seg_path + book + SUFFIX[i], 0, 0)
        with open(part_path + book + SUFFIX[i], 'a+', encoding='utf-8') as f:
            for line in texts:
                if len(line.strip()) > 0:
                    f.write(' '.join(jieba.cut(line.strip())) + '\n')


# 单本实验成功，批量处理十四本的分词任务
def participle_books():
    for book in BOOKS:
        participle_book(book)


# 制作报告用切分及人物提取汇总图表
def result_graph():
    raw = ''.join(MYTOOL.read_txt_file(ROLES, 0, 0)).split("$$$")
    roles = {text.strip().split('\n')[0]: [line.strip().split()[0] for line in text.strip().split('\n')[1:]]
             for text in raw}
    for book in BOOKS:
        texts = MYTOOL.read_txt_file(seg_path + book + SUFFIX[0], 0, 0)
        texts = '\n'.join([line.strip() for line in texts if len(line.strip()) > 0])
        segs = texts.split('$$$')
        # print(book, len(segs))
        print(f"{book}\t{len(segs)}\t{len(roles[book])}\t{'，'.join(roles[book][:4])}")


# 绘制射雕三部曲的词云
def role_cloud():
    raw = ''.join(MYTOOL.read_txt_file(ROLES, 0, 0)).split("$$$")
    roles = {text.strip().split('\n')[0]: [line.strip().split()[:2] for line in text.strip().split('\n')[1:]]
             for text in raw}
    # print(roles)

    fre = roles[BOOKS[4]] + roles[BOOKS[7]] + roles[BOOKS[2]]
    fre = dict([(i[0], int(i[1])) for i in fre])
    del fre['长剑']
    del fre['屠龙刀']
    print(fre)
    backgroud_Image = plt.imread('a.jpg')
    wc = WordCloud(background_color='white',  # 设置背景颜色
                   mask=backgroud_Image,  # 设置背景图片
                   font_path='C:/Windows/fonts/simhei.ttf',  # 设置字体格式，如不设置显示不了中文
                   max_font_size=50,  # 设置字体最大值
                   random_state=30,  # 设置有多少种随机生成状态，即有多少种配色方案
                   )
    wc.generate_from_frequencies(fre)
    plt.imshow(wc)
    plt.axis('off')
    # plt.show()
    plt.savefig('射雕三部曲人物词云.png', dpi=1000)


# 统计两个小数字，用完就over
def numbers():
    N = 0
    for book in BOOKS:
        names = [i.strip().split('\t')[0] for i in MYTOOL.read_txt_file(entity_path + book + SUFFIX[0], 0, 0) if int(
            i.strip().split('\t')[1]) >= 2]
        N += len(names)
    print(f"N={N}")
    raw = [i for i in MYTOOL.read_txt_file(ROLES, 0, 0) if len(i.split()) > 3]
    print(f"n={len(raw)}")


if __name__ == '__main__':
    pass
    # participle_books()
    # participle_book("飞狐外传")
    # result_graph()
    # role_cloud()
    numbers()
