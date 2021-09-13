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

IMPALA marges the learnings acquired from distributed Deep Learning and RL.
In this case we also lack the data buffer and have the separation of actors and learners:
- **Learners**: Implement a parallelised gradient descent mechanism to efficiently update the network weights across multiple machines.
- **Actors**: Can act independently from the learning process and generate samples faster.

In previous approach you first need to generate some data and then wait until the network gets updated, while now this gets decoupled.

**Policy Lag Problem:** Decoupling acting and learning can make the actors follow policies which are quite older than the latest computed by the learners.
This means they produce samples from a different distribution (policy) than the one that will get updated.

**Solution:** V-trace: weight the network updates inversely proportional to the policy distance which generated them. In the [paper](https://arxiv.org/abs/1802.01561), they show how this mitigates the issue.

![](/post/distributed_rl/impala.png)

## 2018. [Ape-X](https://arxiv.org/abs/1803.00933) / [R2D2](https://openreview.net/pdf?id=r1lyTjAqYX)

This method takes a step back into GORILA and uses again the replay buffer mechanism.
Similarly, the actors are separated from the learning process and generate data asynchronously feeding the data points into the replay buffer.
This approach is very scalable, you can have multiple actors sampling independently feeding the buffer.

The main novelty of this work is the sorting of the data in the replay buffer using **distributed prioritization**.
This technique works by setting a priority to each data point fed into the buffer.
This allows the learner to sample from this scoring distribution which should be designed to facilitate the learning process.
For instance you can assign a higher priority to new samples.
Once the learner evaluates a point assigns a lower priority so chances it gets re-sampled are lower.

**Problem**: You end up sampling too much recent data and becomes a bit myopic.

**Solution**: The same actor ANN assigns priorities avoiding the recency bias.

![](/post/distributed_rl/apex.png)

Performance-wise greatly outperformed all other SOTA algorithms by that time.

**OBS**: R2D2 (Recurrent Ape-X algorithm) is essentially the same with an LSTM.

## 2019. Using expert demonstrations [R2D3](https://arxiv.org/abs/1909.01387)

This algorithm uses both expert demonstrations and agent's trajectories.
It sets some sampling probability to each datapoint depending on its origin.

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
