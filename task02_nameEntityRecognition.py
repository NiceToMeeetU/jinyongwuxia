# -*- coding: utf-8 -*-
# @Time    : 20/03/14 14:59
# @Author  : Wang Yu
# @Project : jinyongwuxia
# @File    : task02_nameEntityRecognition.py
# @Software: PyCharm


import os
# import fool
import MYTOOL
import time

all_file = os.listdir("data/金庸原著")


def test1():
    all_file = os.listdir("data/金庸原著")
    for file in all_file:
        start = time.time()
        data = MYTOOL.read_txt_file("data/金庸原著/" + file, 0, 0)
        data = [line.strip() for line in data if len(line.strip()) > 0]
        text = '\n'.join(data)
        res = fool.analysis(text)[1][0]
        element = set([i[2:] for i in res])
        try:
            with open(f"data/命名实体_{len(element)}_" + file, 'a+', encoding='utf-8') as f:
                for i in element:
                    f.write(i[0] + '\t' + i[1].strip() + '\n')
            print(f"{file} \t提取成功，耗时\t{time.time() - start:0.2f}s；")
        except UnicodeEncodeError:
            print(f"{file} 提取失败。")


# 对提取结果初步分析
def check_person():
    data = MYTOOL.read_txt_file("data/命名实体_3370_神雕侠侣（新修版）.txt", 0, 0)
    text = '\n'.join([line.strip() for line in
                      MYTOOL.read_txt_file("data/金庸原著/神雕侠侣（新修版）.txt", 0, 0) if len(line.strip()) > 0])
    person = []
    for line in data:
        i = line.strip().split('\t')
        if i[0] == 'person' and len(i[1]) > 1:
            person.append(i[1])
    # 统计人名出现次数
    D = {name: text.count(name) for name in set(person)}
    # 按照出现次数排序
    D = sorted(D.items(), key=lambda x: x[1], reverse=True)
    # 处理删除不完整名字
    for i in range(4, 50):
        for j in D[i - 4:i + 4]:
            if D[i][0] in j[0] and D[i] != j:
                D.remove(D[i])

    for i in D[:50]:
        print(i[0], i[1])


# 使用foolnltk工具提取命名实体，存入txt文件
def NER_raw():
    # all_file = os.listdir("data/金庸原著")
    for file in all_file:
        start = time.time()
        data = MYTOOL.read_txt_file("data/金庸原著/" + file, 0, 0)
        data = [line.strip() for line in data if len(line.strip()) > 0]
        text = '\n'.join(data)
        res = fool.analysis(text)[1][0]
        element = set([i[2:] for i in res])
        try:
            with open(f"data/命名实体_{len(element)}_" + file, 'a+', encoding='utf-8') as f:
                for i in element:
                    f.write(i[0] + '\t' + i[1].strip() + '\n')
            print(f"{file} \t提取成功，耗时\t{time.time() - start:0.2f}s；")
        except UnicodeEncodeError:
            print(f"{file} 提取失败。")


# 对提取结果初步处理，删除重复名字，统计出现频率
def clean_entity(entity_file, raw_file, category):
    entity0 = MYTOOL.read_txt_file(entity_file, 0, 0)
    raw = MYTOOL.read_txt_file(raw_file, 0, 0)
    text = '\n'.join([line.strip() for line in raw if len(line.strip()) > 0])
    result = []
    for line in entity0:
        i = line.strip().split('\t')
        if i[0] == category and len(i[1]) > 1:
            result.append(i[1])
    # 统计人名出现次数
    result = {name: text.count(name) for name in set(result)}
    # 按照出现次数排序
    D = sorted(result.items(), key=lambda x: x[1], reverse=True)
    # 删除不完整的名字
    for i in range(4, 60):
        for j in D[i - 4:i + 4]:
            if D[i][0] in j[0] and D[i] != j:
                D.remove(D[i])
    return D


# 把所有的初始提取结果清洗后存入新的文件待用
def NER_result():
    # all_file = os.listdir("data/金庸原著/")
    # 选择命名实体类别
    category = 'person'
    for file in all_file:
        cleaned_entity = clean_entity("data/命名实体_" + file, "data/金庸原著/" + file, category)
        with open(f"data/entity/entity_{category}_{file}", 'a+', encoding='utf-8') as f:
            for i in cleaned_entity:
                f.write(str(i[0]) + '\t' + str(i[1]) + '\n')
            print(file, "清洗完成")
    # print(all_file)


# 整合所有命名实体，作为自定义词典用于分词
def merge_entity():
    print(all_file)
    Trilogy = ['射雕英雄传（新修版）.txt', '神雕侠侣（新修版）.txt', '倚天屠龙记（新修版）.txt']
    result = []
    for file in Trilogy:
        entity = MYTOOL.read_txt_file("data/entity/entity_person_" + file, 0, 0)
        for line in entity:
            i = line.strip().split('\t')
            if int(i[1]) >= 100:
                result.append(i[0])
    print('\n'.join(result))


if __name__ == '__main__':
    # NER_result()
    merge_entity()
