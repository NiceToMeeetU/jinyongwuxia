# -*- coding: utf-8 -*-
# @Time    : 20/03/19 11:38
# @Author  : Wang Yu
# @Project : 07 My_NLP
# @File    : test_wordcloud.py
# @Software: PyCharm


import matplotlib.pyplot as plt
import pickle
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import jieba
# from scipy.misc import imread

# 直接从文件读取数据

#读取一个txt文件

text = open('data/participle_金庸原著/倚天屠龙记（新修版）.txt','r',encoding='utf-8').read()

#读入背景图片

bg_pic = plt.imread('a.jpg')

#生成词云

wc = WordCloud(mask=bg_pic,background_color='white',font_path='C:/Windows/fonts/simhei.ttf',scale=1.5).generate(text)

image_colors = ImageColorGenerator(bg_pic)
#显示词云图片

plt.imshow(wc)
plt.axis('off')
plt.show()
