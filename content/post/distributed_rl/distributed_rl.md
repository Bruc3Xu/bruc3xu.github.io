---
title: "Distributed RL"
date: 2020-10-21T09:34:56+08:00
lastmod: 2020-10-30T09:34:56+08:00
draft: false
keywords: []
description: ""
tags: [reinforcement learning]
author: ""

comment: false
toc: true

---
分布式强化学习算法可以大幅提升采样效率，加速学习速度，对于on-policy算法一定程度也能减少方差。
<!--more-->


# History of large scale distributed RL

## 2013. Original [DQN](https://arxiv.org/abs/1312.5602)

2013年DeepMind实现的DQN算法。

![](/post/distributed_rl/dqn.png)

## 2015. General Reinforcement Learning Architecture ([GORILA](https://arxiv.org/abs/1507.04296))

可以分为4个可重复的部分，运行在不同的节点上：
- **replay buffer/memory**: 存储transition$(s, a, r, s^\prime)$
- **learner**: 从replay memory拉取数据更新Q网络
- **actor**: 拉取网络参数，与环境交互得到$(s, a, r, s^\prime)$ 并存入memory buffer
- The **parameter server**: 不断更新保存Q网络的参数

![](/post/distributed_rl/gorila.png)

**Bottleneck**:
actor拉取频率过快影响采样速度。

## 2016. Asynchronous Advantage Actor Critic ([A3C](https://arxiv.org/pdf/1602.01783.pdf))

![](/post/distributed_rl/a3c.png)
所有worker在一台机器上，并通过共享内存分享网络权重。

每个worker采样更新自身的网络，计算梯度并上传。

每个worker利用反馈的总梯度更新网络。

优点：
- 没有网络开销
- 异步更新，采样更快

缺点：
- policy lag：不同worker的网络参数差距过大，总的梯度计算就会不准确。

## 2017. Importance Weighted Actor-Learner Architectures ([IMPALA](https://arxiv.org/abs/1802.01561))

主要包括：
- **Learners**: 并行梯度更新算法，需要等待收集一个batch的样本学习
- **Actors**: 独立于learner，互相不关联，异步更新。

**Solution:** V-trace机制纠正了新旧策略之间的差异，解决了policy lag问题。[paper](https://arxiv.org/abs/1802.01561)

![](/post/distributed_rl/impala.png)

## 2018. [Ape-X](https://arxiv.org/abs/1803.00933) / [R2D2](https://openreview.net/pdf?id=r1lyTjAqYX)

与GORILA类似，replay buffer，actors和learner位于不同的节点，actors异步更新。不同的是提出了
**distributed prioritization**，认为不同样本的学习优先级不同，可以适用于分布式的样本。

![](/post/distributed_rl/apex.png)


R2D2 (Recurrent Ape-X algorithm)使用了LSTM结构。

## 2019. Using expert demonstrations [R2D3](https://arxiv.org/abs/1909.01387)

同时使用专家经验和策略与环境交互数据，并分配不同的比重。

![](/post/distributed_rl/r2d3.png)

## Others:

### [QT-Opt](https://arxiv.org/pdf/1806.10293.pdf)
实际中应用与机器手抓取学习的分布式强化学习架构，提出了一种新的
QLearning算法。

![](/post/distributed_rl/qt_opt.png)


### [Evolution Strategies](https://arxiv.org/abs/1703.03864)
进化算法是一种黑盒优化算法，也是启发式的算法。自然进化中每一代都有突变的基因，环境对突变的基因进行评估，重组产生下一代，直至最优。

其中，关键点在于：
- 基因如何表示（神经网络参数）
- 突变产生（参数优化）
- 基因重组（参数重组）

黑盒优化的优点：
- 不关心奖励分布，奖励密集或稀疏都无所谓
- 不需要反向传播梯度
- 可以适应长期回报，在长动作序列上有优势


![](/post/distributed_rl/evolution.png)

### [Population-based Training](https://deepmind.com/blog/article/population-based-training-neural-networks)
基于种群的训练方法，主要用来自适应调节超参数。

两种常用的自动调参的方式是并行搜索(parallel search)和序列优化(sequential optimisation)。并行搜索就是同时设置多组参数训练，比如网格搜索(grid search)和随机搜索(random search)。序列优化很少用到并行，而是一次次尝试并优化，比如人工调参(hand tuning)和贝叶斯优化(Bayesian optimisation)。并行搜索的缺点在于没有利用相互之间的参数优化信息。而序列优化这种序列化过程显然会耗费大量时间。

文章提出将并行优化和序列优化相结合。既能并行探索，同时也利用其他更好的参数模型，淘汰掉不好的模型。

![](/post/distributed_rl/population.png)


如图所示，(a)中的序列优化过程只有一个模型在不断优化，消耗大量时间。(b)中的并行搜索可以节省时间，但是相互之间没有任何交互，不利于信息利用。(c)中的PBT算法结合了二者的优点。

首先PBT算法随机初始化多个模型，每训练一段时间设置一个检查点(checkpoint)，然后根据其他模型的好坏调整自己的模型。worker会定期exploit种群其他的模型参数，得到更优的模型参数，并添加随机扰动再进行explore。其中checkpoint的设置是人为设置每过多少step之后进行检查。扰动要么在原超参数或者参数上加噪声，要么重新采样获得。

![](/post/distributed_rl/population_code.png)

