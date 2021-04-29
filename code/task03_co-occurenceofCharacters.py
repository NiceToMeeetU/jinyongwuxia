# -*- coding: utf-8 -*-
# @Time    : 20/03/14 16:11
# @Author  : Wang Yu
# @Project : jinyongwuxia
# @File    : task03_co-occurenceofCharacters.py
# @Software: PyCharm


"""
文档已按章节切分，按照给定的自定义词典分词，所有主要人物筛选完毕
开始做人物贡献分析及对比
运行前需保证由 task01_textSegmentation.py 得到的28篇文档和主要人物表在data/内
"""

import MYTOOL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline, interp1d
import jieba
from scipy.stats import entropy

# 分好词的完整文档路径
part_path = "../data/participle_金庸原著/"
# 所有书名列表
BOOKS = ['书剑恩仇录', '侠客行', '倚天屠龙记', '天龙八部', '射雕英雄传', '白马啸西风', '碧血剑',
         '神雕侠侣', '笑傲江湖', '连城诀', '雪山飞狐', '飞狐外传', '鸳鸯刀', '鹿鼎记']
# 两种不同的后缀
SUFFIX = ["（新修版）.txt", "（三联版）.txt"]
# 人物表
ROLES = "../data/14本书主要人物出现次数（清理）.txt"


# ★★★★★ 查找目标词在语句中窗口的索引位置
def word_window_index(word_in, list_in, window_size_in):
    """
    查找目标词在单个语句中的所有窗口，及窗口中心位置
    :param word_in: (str)单个单词
    :param list_in: (list)单句文档，完成分词
    :param window_size_in: (int)窗口大小
    :return result_out: (list)所有窗口索引在原序列中的位置的列表
            center_index: (list)每个窗口的中心在原序列的位置的列表
    """
    result_out = []
    center_index = []
    loc0 = -1
    list_len = len(list_in)
    while True:
        try:
            loc0 = list_in.index(word_in, loc0 + 1)
            center_index.append(loc0)
            window_index_out = list(set(range(loc0 - window_size_in, loc0 + 1 + window_size_in))
                                    & set(range(list_len)))
            result_out.append(window_index_out)

        except ValueError:
            break

    return result_out, center_index


# ★★★★★ 统计目标词组在语料库中上下窗口内的总词数
def window_count_score(words_in, texts_in, window_size_in):
    """
    统计目标词组
    :param words_in: (list)目标词列表
    :param texts_in: (list)语料库列表
    :param window_size_in: (int)窗口大小
    :return:
    """
    length = len(words_in)
    result_out = np.zeros(length)
    # 以1/D作为权值计算出现分数
    score = [1 / float(n) for n in range(1, window_size_in + 1)]
    for text_in in texts_in:
        for i in range(length):
            window_one, _ = word_window_index(words_in[i], text_in, window_size_in)
            for each_window in window_one:
                window_len = len(each_window)
                result_out[i] += sum(score) + sum(score[:window_len - 1 - window_size_in])
    return result_out


# ★★★★★ 统计目标词组互相在窗口内出现的次数
def co_occurrence_score(words_in, texts_in, window_size_in):
    """
    统计目标词组互相在彼此窗口内出现的次数
    :param words_in: (list)目标词组列表
    :param texts_in: (list)语料库列表，格式：两层列表，外层按段落分，内层是分词结果按空格分
    :param window_size_in: (int)窗口大小
    :return: (mat)词-词共现矩阵
    """
    length = len(words_in)
    result_out = np.zeros((length, length))
    for text_in in texts_in:
        for i in range(length):
            # 找词袋中的第i个词在语句text_in中的所有窗口及窗口中心
            window_one, center_one = \
                word_window_index(words_in[i], text_in, window_size_in)
            for k in range(len(window_one)):
                # 对第k个窗口再做最内层循环
                for j in range(length):
                    loc0 = -1
                    while True:
                        try:
                            # 在该第i个词在text_in句内的第k个窗口中从头查询词袋中的第k个词，直到找不到
                            loc0 = text_in.index(words_in[j], max(window_one[k][0], loc0 + 1), window_one[k][-1] + 1)
                            # 找到第k个词后，只要其不是中心位置（即不是i词本身），则计算其分数
                            if loc0 != center_one[k]:
                                result_out[i][j] += 1 / float(abs(loc0 - center_one[k]))
                        except ValueError:
                            break
    return result_out


# ╳╳╳╳╳ 计算整篇文档各部分的目标人物共现矩阵
def cal_co_mat(file_seg_in, people, window):
    """
    计算整篇文档各部分的目标人物共现矩阵
    :param file_seg_in: (str)需要统计的文件地址，应当是已用空格分好词的文件
    :param people: (list)要研究的共现人物列表，一般把主人公放在首位
    :param window: (int)计算共现的窗口大小
    :return:(matrix)people中所有人物在主人公的窗口出现的频率矩阵，i-章节数，j-人物，
    """
    # texts = divide_list(MYTOOL.read_txt_file(file_seg_in, 0, 0), parts)
    texts = MYTOOL.read_txt_file(file_seg_in, 0, 0)
    texts = '\n'.join([line.strip() for line in texts if len(line.strip()) > 0])
    # segs = texts.split('$ $ $')
    segs = [i.strip().split('\n') for i in texts.split('$ $ $')]

    print(len(segs))
    result = []
    for one_part in segs:  # one_part list
        # 将完整文档每行按空格切开存成列表
        s = [line.strip().split() for line in one_part]
        girls = co_occurrence_score(people, s, window)
        # 共现矩阵各元素除以第一行共现次数之和
        # 即统计所有人在第一个人周围出现的频率
        # 暂时先以主人公视角来看问题
        if girls.sum(axis=1)[0] != 0:
            result.append(girls[0] / girls.sum(axis=1)[0])
        else:
            result.append(np.zeros(len(people)))
    return np.array(result)


# ★★★★★ 指数平滑
def exponential_smoothing(s_in, alpha=0.5):
    """
    指数平滑公式
    :param s_in: 待平滑序列
    :param alpha: 平滑指数
    :return: 指数平滑完毕的序列
    """
    s_out = np.zeros(s_in.shape)
    s_out[0] = s_in[0]
    for i in range(1, len(s_out)):
        s_out[i] = alpha * s_in[i] + (1 - alpha) * s_out[i - 1]
    return s_out


# ╳╳╳╳╳ 列表均分成若干份
def divide_list(l_in, parts):
    """
    列表均分成若干份
    :param l_in: (list)待分割的列表
    :param parts: (int)需要平分成的份数
    :return: (list)切分好的列表
    """
    p = int(len(l_in) / parts)
    l_out = [l_in[i:i + p] for i in range(0, (parts - 1) * p, p)]
    l_out.append(l_in[(parts - 1) * p:])  # 最后一部分全部算在一起
    return l_out


# 绘制共现频率曲线，颜色，线形，画幅待优化
def draw(mat_in, labels_in, smooth_n, alpha, show_flag=True, save_flag=False, name=None):
    """
    绘制共现频率曲线
    :param mat_in: (matrix)需要绘制的共现频率矩阵数据，i=章节数即文档切分数，j=总人物数-1
    :param labels_in: (list)人名列表，长度为总分析人数-1 ，即原共现矩阵维度-1
    :param smooth_n: (int)进行三次样条插值的加密点数
    :param alpha: (float,0,1)指数平滑系数
    :param show_flag: (bool)是否显示图像
    :param save_flag: (bool)是否保存图像
    :param name: 可能必要的存储名字
    :return: none
    """
    x = np.arange(len(mat_in))
    x_smooth = np.linspace(x.min(), x.max(), smooth_n)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['Simhei']
    plt.rcParams['axes.unicode_minus'] = False
    # plt.figure(figsize=(16, 8), dpi=400)
    for i in range(len(labels_in)):
        y = make_interp_spline(x, exponential_smoothing(mat_in[:, i], alpha))(x_smooth)
        plt.plot(x_smooth, y, label=labels_in[i])
    plt.legend(labels_in, loc=0, ncol=2, prop={'size': 18})
    plt.xlabel('章节数', fontdict={'weight': 'normal', 'size': 18})
    plt.ylabel('人物共现频率', fontdict={'weight': 'normal', 'size': 18})
    plt.tick_params(labelsize=18)
    # plt.figure(dpi=2000)

    if show_flag:
        plt.show()
    if save_flag:
        plt.savefig(f"img/{labels_in[0]}{name}.png")


#  画多幅子图
def mini_draw(p, q, roles, smooth_n=400, alpha=0.5):
    x = np.arange(len(q))
    x_smooth = np.linspace(x.min(), x.max(), smooth_n)
    plt.close()
    plt.rcParams['font.sans-serif'] = ['Simhei']
    plt.rcParams['axes.unicode_minus'] = False
    labels = ["新修版", "三联版"]
    plt.legend(labels)

    for i in range(4):
        y0 = make_interp_spline(x, exponential_smoothing(p[:, i + 1], alpha))(x_smooth)
        y1 = make_interp_spline(x, exponential_smoothing(q[:, i + 1], alpha))(x_smooth)
        plt.subplot(221 + i)
        plt.plot()
        plt.plot(x_smooth, y0, label=labels[0])
        plt.plot(x_smooth, y1, label=labels[1])
        # plt.legend(labels)
        plt.title(roles[i + 1], fontdict={'weight': 'normal', 'size': 13})
        plt.xticks([])
        plt.yticks([])
    # plt.subplot(111)

    plt.show()

    # t1 = np.arange(0, 5, 0.1)
    # t2 = np.arange(0, 5, 0.02)
    # # plt.figure(12)
    # plt.subplot(221)
    # plt.plot(t1, np.exp(-t1) * np.cos(2 * np.pi * t1), 'bo')
    #
    # plt.subplot(222)
    # plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
    #
    # plt.subplot(223)
    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    #
    # plt.subplot(224)
    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    #
    # plt.show()


# 计算主角共现频率矩阵对应KL散度的范数
def KL_dis(p, q):
    return np.linalg.norm(entropy(p, q + 0.00000001))


# 如果不算KL散度，直接求差呢？
def sub(p, q):
    return np.linalg.norm(p - q)


# ★★★★★从主要人物表中分割提取各部书的人物，形成字典便于后续使用
def extract_roles():
    """
    从处理好的主要人物txt里提取人物信息
    :return: 字典，k=书名，v=(list)角色名按序排布
    """
    raw = ''.join(MYTOOL.read_txt_file(ROLES, 0, 0)).split("$$$")
    roles = {text.strip().split('\n')[0]: [line.strip().split()[0] for line in text.strip().split('\n')[1:]]
             for text in raw}
    return roles


# ★★★★★分开文档
def read_by_section(file):
    """
    将文档完全分开
    1）按$$$分成各个章节
    2）按\n分成各行
    3）按空格在隔行中分开各词
    :param file: 需要分割的文件地址
    :return: 三维列表，i=章节数，j=章节内行号，k=行内第k个字
    """
    texts = MYTOOL.read_txt_file(file, 0, 0)
    texts = '\n'.join([line.strip() for line in texts if len(line.strip()) > 0])
    segs = [i.strip().split('\n') for i in texts.split('$ $ $')]
    s = [[i.strip().split() for i in j] for j in segs]

    return s


# ★★★★★最终版的最后计算共现频率矩阵
def final_cal(seg_text_in, role, window):
    result = []
    for section in seg_text_in:
        mat = co_occurrence_score(role, section, window)
        if mat.sum(axis=1)[0] != 0:
            result.append(mat[0] / mat.sum(axis=1)[0])
        else:
            result.append(np.zeros(len(role)))
    return np.array(result)


# 单独给倚天开一个函数处理
def yitian():
    pass
    role_yitian = ['张无忌', '赵敏', '周芷若', '殷离', '小昭']
    t0 = read_by_section(part_path + BOOKS[2] + "- 副本" + SUFFIX[0])
    p0 = final_cal(t0, role_yitian, 30)
    t1 = read_by_section(part_path + BOOKS[2] + "- 副本" + SUFFIX[1])
    p1 = final_cal(t1, role_yitian, 30)
    # draw(p0[:,1:],role_yitian[1:],400,0.5,1,0)

    mini_draw(p0, p1, role_yitian)
    #
    # # print('沿轴 1 堆叠两个数组：')
    # # for i in range(40):
    # #     print(p0[i][1], p1[i][1])
    # zhou = np.stack((p0[:, 4], p1[:, 4]), 1)
    # print(zhou.shape)
    # print(len(zhou))
    # print(zhou)
    # mat_in = zhou
    # labels_in = ["新修版", "三联版"]
    # alpha = 1
    # smooth_n = 200
    #
    # x = np.arange(len(mat_in))
    # # x_smooth = np.linspace(x.min(), x.max(), smooth_n)
    # plt.close()
    # plt.rcParams['font.sans-serif'] = ['Simhei']
    # plt.rcParams['axes.unicode_minus'] = False
    # # plt.figure(figsize=(16, 8), dpi=400)
    # # y0 = make_interp_spline(x, exponential_smoothing(mat_in[:, 0], alpha))(x_smooth)
    # # y1 = make_interp_spline(x, exponential_smoothing(mat_in[:, 1], alpha))(x_smooth)
    # y0 = exponential_smoothing(mat_in[:, 0], alpha)
    # y1 = exponential_smoothing(mat_in[:, 1], alpha)
    # plt.plot(x, y0, label=labels_in[0])
    # plt.plot(x, y1, label=labels_in[1])
    # plt.legend(labels_in, loc=0, ncol=2, prop={'size': 18})
    # plt.xlabel('章节数', fontdict={'weight': 'normal', 'size': 18})
    # plt.ylabel('人物共现频率', fontdict={'weight': 'normal', 'size': 18})
    # plt.tick_params(labelsize=18)
    # plt.show()
    # # draw(zhou, ["新修版", "三联版"], 400, 0.5, 1, 1)
    # # print('沿轴 1 连接两个数组：')
    # # print(np.concatenate((p0[:, 2], p1[:, 2]), axis=1))
    # # draw(p_yitian[:, 1:], role_yitian[1:], 400, 0.5)


def yitians():
    role_yitian = ['张无忌', '赵敏', '周芷若', '殷离', '小昭', '谢逊', '范瑶', '韦一笑', '杨逍', "灭绝师太", "宋青书"]
    # role = extract_roles()[BOOKS[2]]
    # print(role)
    for i in [0, 1]:
        for p in [30]:
            # texts_yitian = read_by_section(part_path + BOOKS[2] + "- 副本" + SUFFIX[i])
            texts_yitian = read_by_section(part_path + BOOKS[2] + SUFFIX[i])
            p_yitian = final_cal(texts_yitian, role_yitian, p)
            for q in [0.5, 0.75, 1]:
                draw(p_yitian[:, 1:5], role_yitian[1:5], 400, q, 0, 1, f"raw_窗口{p}_alpha{q}_{i}")


# ★★★★★求单本书的数据
def one(i, edition):
    """
    对单本数的数据统计
    :param i: (int)BOOKS中的索引号
    :param edition: (int)版本号，0是新修版，1是三联版
    :return: (mat)共现频率矩阵，i=章节数，j=人物数
    """
    texts = read_by_section(part_path + BOOKS[i] + SUFFIX[edition])
    role = extract_roles()[BOOKS[i]]
    p = final_cal(texts, role, 20)
    return p


# ★★★★★测试完整28本对比差异度，完成
def compare():
    # result = []
    # for i in range(14):
    #     p0 = one(i, 0)
    #     p1 = one(i, 1)
    #     result.append(sub(p0, p1))
    # print(result)
    a = [0.30193287594296664, 2.4849079486582313, 1.0754849561899182, 1.033224707561813, 0.3764095165505632,
         0.19101364085916545, 0.6040824109876589, 0.49544617674561736, 0.8932877170615181, 0.28708101328317465,
         0.1544900162828267, 0.4290256525539736, 0.023207216980167696, 0.25627525335671986]
    b = [(BOOKS[i], a[i]) for i in range(14)]
    c = sorted(b, key=(lambda x: x[1]), reverse=False)
    print(c)
    for i in c:
        print(i[0], i[1])
    # k = [i[0] for i in c]
    # v = [i[1] for i in c]
    # plt.rcParams['font.sans-serif'] = ['Simhei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.figure(figsize=(3,8),dpi=100)
    # plt.barh(k, v)  # 横放条形图函数 barh
    # plt.tick_params(labelsize=24)
    #
    # plt.show()


def last():
    i = 1
    roles = extract_roles()[BOOKS[i]][:5]
    t0 = read_by_section(part_path + BOOKS[i] + SUFFIX[0])
    p0 = final_cal(t0, roles, 20)
    t1 = read_by_section(part_path + BOOKS[i] + SUFFIX[1])
    p1 = final_cal(t1, roles, 20)
    # draw(p0[:,1:],role_yitian[1:],400,0.5,1,0)

    mini_draw(p0, p1, roles)


if __name__ == '__main__':
    pass
    # pro_one(3,0)
    # compare()
    # yitian()
    # mini_draw()
    last()