# -*- coding: utf-8 -*-
# @Time    : 20/03/12 22:33
# @Author  : Wang Yu
# @Project : 07 My_NLP
# @File    : MYTOOL.py
# @Software: PyCharm

import chardet


# 读取文本文件
def read_txt_file(file_name_in, lines_number_in=0, print_flag=False):
    """读取txt内容并打印输出前若干行"""
    # with open(file_name_in, 'rb') as f0:
    #     s = f0.read()
    # e = chardet.detect(s)['encoding']
    try:
        with open(file_name_in, 'r', encoding='utf-8') as f1:
            lines1 = f1.readlines()
    except UnicodeDecodeError:
        with open(file_name_in, 'r', encoding='gb18030') as f1:
            lines1 = f1.readlines()
    if lines_number_in == 0:
        lines_number_in = len(lines1)

    if print_flag:
        for i in range(lines_number_in):
            print(lines1[i])
    lines_out = lines1[0:lines_number_in]
    return lines_out


def write_txt_file(file_name_in, data_in):
    with open(file_name_in, 'a+') as f:
        for i in data_in:
            f.write(str(i))


# OK——正则表达式去除标点符号
def remove_biaodian(string_in, re_base="[^0-9A-Za-z\u4e00-\u9fa5]"):
    """正则表达式去除标点符号"""
    return re.sub(re_base, '', string_in)


# OK——正则表达式去除标点符号
def zhengze(string_in, re_base="[^，。？\u4e00-\u9fa5]"):
    """正则表达式去除标点符号"""
    return re.sub(re_base, '', string_in)
