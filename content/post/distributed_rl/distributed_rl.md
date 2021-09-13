---
title: "Distributed RL"
date: 2020-10-21T09:34:56+08:00
lastmod: 2020-10-30T09:34:56+08:00
draft: false
keywords: []
description: ""
tags: [reinforcement learning]
categories: [cs285]
author: ""

comment: false
toc: true

---

<!--more-->

分布式强化学习算法可以大幅提升采样效率，加速学习速度，对于on-policy算法一定程度也能减少方差。

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

This distributed architecture main feature is that it interfaces with the real world.
Moreover, its weight update happens asynchronously.
Meaning it can be easily heavily parallelized into multiple cores (can be **independently scaled**).
It was designed for robotic grasping, using a setup of 7 robots creating samples.

![](/post/distributed_rl/qt_opt.png)


### [Evolution Strategies](https://arxiv.org/abs/1703.03864)

Gradient-free approach by OpenAI.
Essentially uses an evolutionary algorithm on the ANN weights.
It works by having  multiple instances of the network where it applies some random noise.
The idea is then to run the policies and perform a weighted average parameter update based on the performance of each policy.

![](/post/distributed_rl/evolution.png)

### [Population-based Training](https://deepmind.com/blog/article/population-based-training-neural-networks)

Technique for hyperparameter optimization.
Merges the idea of a grid search but instead of the networks training independently, it uses information from the rest of the population to refine the hyperparameters and direct computational resources to models which show promise.

Using this technique one can improve the performance of any hyperparam-dependent algorithm.
