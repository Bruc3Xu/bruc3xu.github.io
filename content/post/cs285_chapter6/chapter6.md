---
title: "cs285 DRL notes chapter 6: Deep RL with Q-Functions"
date: 2020-09-18T15:46:38+08:00
lastmod: 2020-09-18T15:46:38+08:00
draft: false
keywords: []
description: ""
tags: []
categories: []
author: ""

# You can also close(false) or open(true) something for this content.
# P.S. comment can only be closed
comment: false
toc: true
autoCollapseToc: false
postMetaInFooter: false
hiddenFromHomePage: false
# You can also define another contentCopyright. e.g. contentCopyright: "This is another copyright."
contentCopyright: false
reward: false


# You unlisted posts you might want not want the header or footer to show
hideHeaderAndFooter: false

# You can enable or disable out-of-date content warning for individual post.
# Comment this out to use the global config.
#enableOutdatedInfoWarning: false

flowchartDiagrams:
  enable: false
  options: ""

sequenceDiagrams: 
  enable: false
  options: ""

---

<!--more-->
# Deep Q Network
- fitted q iteration算法：当前策略收集整个数据集，然后对Q函数进行多次回归近似，接下来收集新的数据集循环这一过程。
- Q-Learning（online q iteration）：一边收集数据，一边进行学习。
## Q-Learning的问题
### Q-learning is not GD
Q-learning不是梯度下降，公式
$$
\begin{aligned}
y_i&=r(s_i,a_i)+\gamma max_{a'}Q_\phi(s_i',a_i')\\\\
\phi&\leftarrow \phi-\alpha{dQ_\phi\over d\phi}(s_i,a_i)(Q_\phi(s_i,a_i)-y_i)  
\end{aligned}
$$
其中$y_i$与参数$\phi$相关，但没有梯度计算，因而不保证收敛。

### one-step gradient
Q-Learning使用一个样本来更新梯度，更新非常不稳定，使用批次更新最优。

### moving targets
目标值因为与神经网络参数相关，每次神经网络更新，目标值也会随之变动，严重影响收敛。

### correlate samples
我们使用神经网路来近似Q函数，而神经网络学习数据一般要求独立同分布（i.i.d），但是在rl的采样中，序列决策问题的数据来源于与环境的相邻交互，数据之间往往是具有相关性的，即可能状态相似。神经网络会在局部状态过拟合，当学习新的状态时，遗忘旧的内容。

## replay buffers
Replay Buffer是一个或者多个智能体采样得到样本的合集，会将所有样本都存储起来，直至容量上限。
当使用Q-Learning算法学习时，从其中随机拿出部分样本来进行神经网络训练。
Replay Buffer解决了上面提到的两个问题：one-step gradient和correlate samples。

使用Replay Buffer的Q-Learning算法如下
***
使用某种策略收集transition$\{(s_i,a_i,s'\_i,r_i)\}$并存储到replay buffer $\mathcal{B}$\
For $K$ times\
    $\quad \text{获取样本批次}(s_i,a_i,s'\_i,r_i)\leftarrow \mathcal{B}$\
    $\quad y_i\leftarrow r(s_i,a_i) + \gamma \max_{a'_i}Q_\phi(s'_i,a'_i)$\
    $\quad \phi \leftarrow \phi-\alpha\Sigma_i\frac{dQ_\phi}{d\phi}(s_i,a_i)(Q_\phi(s_i,a_i) - y_i)$
***
我们需要关心的是
- replay buffer的填充频率
- 存储transition的数量
- 采样的方式（均匀、权重等）

## target network
我们希望目标值能够固定，一种方式是使用Q神经网络，固定其参数，用来计算目标值。目标网络会在神经网络更新后每隔一段时间进行更新，然后重复这一过程。

## general view of DQN