# 金庸全集的新修版和三联版，变动有多大？

> 《文本分析》课程的大作业，从统计学的角度探讨文学作品的人物关系变动，当时全身心投入做得好开心，一直忘了发上来。
>
> 详细代码见Repo，感谢star~



## 1	背景

“飞雪连天射白鹿，笑书神侠倚碧鸳”，半个多世纪来，金庸先生的十四部著作深深影响了整个华语世界，为无数华人营造了波澜壮阔的武侠江湖梦。这套丛书前后先后曾有过三个版本：早期的连载版，94年的三联版和08年的新修版。其中以多次被搬上银屏的三联版更为国人所熟知，尤其笔者这一代人曾通读过的也多为经典的三联版。而金庸先生在耄耋之年，仍笔耕不辍，对所有作品进行了修改更订，想必是对武侠又有了更新的认识和理解。那么这些更新版本中，到底修改了哪些内容，哪部作品又变动幅度最大呢？

本文从统计学视角着力，利用文本分析技术，量化构建一种通用模型来反映文学作品内容层面的变更，对金庸作品集三联和新修两个版本做深层次的可视化对比，分析整体修订的特点，并推广到其他作品。

总的来说，实验主要解决三个方面问题：

1） 在对文本没有了解的情况下，如何快速批量提取合理的文本特征；

2） 如何构建合理的特征映射，得到能反映文本内容的量化指标；

3） 如何将枯燥的文本分析内容进行合理的可视化呈现。

本次实验基于python3.7完成，数据来源为网络搜集的金庸作品集三联版和新修版合计28个txt文档。



## 2	算法模型

### 2.1	解决思路

人在阅读文学作品时，一般是跟随主角的视角参与到故事的发展，通过角色的对话、互动串联起完整的情节。可以说人物形象是文学作品内容的核心，那么掌握了角色人物间的变化，也就能了解到故事内容的变化。

那么如何用量化的方式分析人物关系变动？著名的词向量嵌入算法`GloVe`中，通过构建词-词共现矩阵，以共现频率之比表征词与词之间的关系，进而构造出稠密向量完成单词的高维嵌入，能有效地反映文本全局的词频信息，高效完成单词向量化。

本次实验即受这种思想启发：若文学作品中，两个人物频繁共同出现，必定说明两者有较强的关系，这种关系即能直接反映故事情节的发展。所以以人物为着眼点，统计不同人物在彼此上下文窗口中出现的次数，加权求和后再比上窗口内人物总数，即得人物的共现频率。如果同一篇章内，相同的两个任务的共现频率发生变化，即说明文学作品内容发生了实质性变化。

### 2.2	模型设计

对不同版本的同一作品$T_1, T_2$，对其以任务词典分词后，按照同样的章节架构切分成$M$节，每节记为$S_k,\ k=1,2,\ldots M$。

该作品的角色提取结果共有$N$个，将其按出现频次高低排序，取其中$n$个研究，每个角色记为$r_i,\ i=1,2,\ldots n$。合理假设出现频词最高的$r_i$为作品主角。在$s_k$节中，统计各角色在彼此窗口上下文中出现的次数。

以角色 $r_i$ 为中心，长度为 $2 \times a $ 构建出 $W_i$ 个窗口，在某个窗口 $w_l (l = 1,2,...,W_i)$ 中，若角色$r_j$  出现且距窗口中心距离为 $d_{ilj}$ ，则以距离长度反比做权重统计该窗口内共现次数，累加所有窗口统计结果即得两角色 $r_i r_j$ 在本节中的共现矩阵 $X_{S_{k}}$ ：
$$
{X_{S_k}=\ \left[\begin{matrix}&\vdots&\\\cdots&x_{ij}&\cdots\\&\vdots&\\\end{matrix}\right]}_{n\times n}
$$

$$
{\ x}_{ij}=\sum_{l=1}^{W_i}\frac{1}{d_{ilj}},\ \ {\ x}_i=\sum_{j=1}^{n}\sum_{l=1}^{W_i}\frac{1}{d_{ilj}}
$$

其每个元素 $x_{ij}$ 表示角色 $r_i$ 在角色 $r_j$ 窗口中出现的加权次数，$x_i$  为该共现矩阵的行求和，即角色 $r_i$ 周围出现的人物总数。定义共现频率表征角色 $r_i$ 在角色 $r_j$ 窗口中出现的频率。
$$
{\ p}_{ij}=\frac{{\ x}_{ij}}{{\ x}_i}
$$
当  取1，  取1到n时，即表示所有其他人物与主角  的共现频率，反映人工阅读文章时的直观感受。

矩阵  表示版本  的主角共现频率，按列求KL散度后再按行求范数即得两版本内容变化评分函数，以此作为模型目标函数：

 

几点说明：

1） 统计角色共现矩阵时按照章节拆分统计，是为了增加数据的颗粒度，避免全局通盘考虑抹除了差异，提高数据准确性；

2） 只统计共现矩阵  的第一行，即只考虑主角与其他角色的共现频率。因为  是高度稀疏的，部分次要角色无共现，其余行数据包含信息有限，为减少计算复杂度，只统计角色与主角的共现；

3） 该模型设计后在《倚天屠龙记》中做了验证，与笔者真实阅读感受及网络上相关参考信息相符，模型被证明有效后才推广到所有作品集计算。

# 1  实验过程

## 1.1 命名实体识别

本次实验主要采用Hanlp工具包和foolNLTK工具包进行命名实体识别，再进行交叉验证。两个工具包都是基于tensorflow搭建的深度神经网络完成的识别任务。

实验过程中发现存在的问题和解决方法如下：

### 1)  部分动词无法分开：

金庸作品文白参杂，有大量类似“杨过道”，“赵敏道”的短语，两个工具包都将此直接识别成了人名，若不处理则会给后续分词工作造成了很大影响。

解决方法，提前构建小型自定义词典，包含“道”、“闻”、“想”等通过初步识别发现的常见错误项，使用分词工具进行分词，保证时“人名-道”这种词组能被优先分开，提高人名识别正确率；

### 2)  部分常见词组被误识别：

诸如“武林”、“朗声道”等与人名结构相似词，“长剑”、“武当”、“那人”等高频词极易被误识别为人名，实际则未承担角色功能。

解决方法：对所有作品的人名集合求交集，对其中共同多次出现的明显不是角色名的词组进行剔除。（此处应用先验知识，需将《射雕英雄传》、《神雕侠侣》、《倚天屠龙记》射雕三部曲和《雪山飞狐》、《飞狐外传》分别看作整体处理）；

### 3)  角色变更问题：

统计主要人角色出现频次后，结合原著后记中作者的相关说明发现，部分作品新修后完全修改了个别角色姓名，如金轮法王全部改为金轮国师，尹志平全部改为甄志炳等，因为涉及到后续的共现统计，所以对于此类明确的同一人物不同姓名的变更，以三联版为准替换了新修版，达成统一。

## 1.2 文档切分和分词

本次实验原始数据是28部作品的txt格式文本，全文存储在一个文件中，且章回标题各不相同，包括“第一回”，“第一章”，“一”等多种形式。实验利用正则表达式分类匹配，交叉验证，将不同版本的各部作品按拆分为相同的章节结构。

利用3.1得到的主要角色频次表（见附件），构建自定义词典，加载进jieba工具对文档进行分词处理，保证所有主要角色姓名都被完整保留。

## 1.3 角色共现统计

统计角色共现的窗口类似于卷积操作，窗口不跨行统计。

该部分程序输入是三层嵌套的列表，从外到内依次是章节号，行号，行内序号，最内层的列表内是单个的分好词的字符串词组，该部分代码有较强的鲁 棒性，能批量循环在章节内构建窗口，高效完成汇总任务，是本次实验的难点之一.

## 1.4 数据可视化

基于matplotlib模块和wordcloud模块完成数据可视化，绘制章节-共现频率曲线是，为提升美观度，使用了三次样条插值和指数平滑。

# 2  结果分析

## 2.1 角色提取结果

经初次识别和分词清洗后二次识别，共在28部作品中抽取到8228个人名实体，进行一定程度的手工筛选后，选取出现频次大于100的合计502个人名进行研究，详见表1。举例对神雕三部曲中提取的人名绘制词云如图1。

 

图1 《射雕三部曲》人物词云示意

可以看到，三部曲的男主角大小有明显差异，郭靖虽然跨越两部作品，但由于《射雕英雄传》是系列第一部，承担着世界观设定的任务，男主起到的情节推动不大；第二部《神雕侠侣》全程基本上单线叙事，杨过的视角占到了绝大篇幅；而第三部《倚天屠龙记》中张无忌到全书四分之一的地方才出场，却仍有与杨过相当的频次，可见本书主角出现频次极高。又因为笔者通读了《倚天》三联版和新修版，故以此作品来检验模型有效性。

## 2.2 文档切分结果

利用正则语法，将所有作品按照章节切分，形成可用于快速测试的可迭代对象，相关结果见表1。

表1 金庸作品集章节数及主要角色数

| 书名       | 章节 | 角色 | 主要角色示例                   |
| ---------- | ---- | ---- | ------------------------------ |
| 书剑恩仇录 | 20   | 37   | 陈家洛，张召重，徐天宏，霍青桐 |
| 侠客行     | 21   | 19   | 石破天，石清，丁珰，白万剑     |
| 倚天屠龙记 | 40   | 56   | 张无忌，赵敏，谢逊，张翠山     |
| 天龙八部   | 50   | 71   | 段誉，萧峰，慕容复，虚竹       |
| 射雕英雄传 | 40   | 51   | 郭靖，黄蓉，欧阳锋，洪七公     |
| 白马啸西风 | 9    | 6    | 李文秀，苏普，阿曼，苏鲁克     |
| 碧血剑     | 20   | 42   | 袁承志，青青，何铁手，袁崇焕   |
| 神雕侠侣   | 40   | 48   | 杨过，小龙女，郭靖，黄蓉       |
| 笑傲江湖   | 40   | 41   | 令狐，岳不群，盈盈，林平之     |
| 连城诀     | 12   | 13   | 狄云，水笙，戚芳，丁典         |
| 雪山飞狐   | 10   | 11   | 曹云奇，胡斐，宝树，苗若兰     |
| 飞狐外传   | 20   | 29   | 胡斐，程灵素，袁紫衣，苗人凤   |
| 鸳鸯刀     | 1    | 7    | 萧中慧，袁冠南，卓天雄，周威信 |
| 鹿鼎记     | 50   | 70   | 韦小宝，皇上，康熙，皇帝       |

## 2.3 《倚天屠龙记》人物分析

为检测模型有效性，对《倚天屠龙记》三联版和新修版进行数据测试。根据先验信息，《倚天》是一男主四女主设置，五人间关系错综复杂，故重点分析这五个角色的共现频率，验证模型。

以新修版示例，取窗口大小为30计算上述四位女主角与男主的共现频率，加密点设为400，平滑系数设为0.5，绘制章节-共现频率曲线如图2。

可以看到，曲线较清晰地反映了四位女主角出现的时间和强度，体现了男主角与其的交互程度。以周芷若举例，其首次出现在第10节，书中内容此处为其在汉水舟中给张无忌喂饭，两人有较强烈的互动，而后两人分离，直到第34节新妇素手裂红裳才与张无忌结婚而有较大的互动，之后出走后复在少林大会现身，与张无忌的关系经历了巨大的起伏，曲线能准确地反映情节变化。



