---
title: "Soft Actor Critic"
date: 2020-11-13T09:34:56+08:00
lastmod: 2020-11-13T09:34:56+08:00
draft: false
keywords: []
description: ""
tags: [reinforcement learning]
author: ""

comment: false
toc: true

---

<!--more-->


## Idea
现有深度强化学习算法主要的问题有：
 - 采样难，样本利用率低：对于一般的强化学习问题，学习得到想要的策略需要的样本以百万、千万记，而绝大多数on-policy算法在策略更新后丢弃旧的样本。
 - 训练不稳定，收敛困难：off-policy目的在于利用不同于当下策略采集的样本来学习。

Soft Actor Critic是基于最大熵框架的off-policy、actor-critic算法。

### Maximum Entropy Framework
*"Succeed at the task, while behaving as random as possible"*

Actor在最大化期望奖励的同时最大化熵：
- 更多的exploration避免遗漏更好的trajectory。
- 学习得到的策略更加鲁棒，因为学习过程中尽可能采取随机的策略（可以看做噪声更多），那么得到的策略在实际测试中会更加具有泛化性。

优化目标变为：

$$
J(\pi) = \sum_t E_{(s_t, a_t) \sim \rho_{\pi}} \left[ r(s_t, a_t) + \alpha H(\pi(\cdot | s_t)) \right]
$$

其中$\alpha$是"temperature"参数，可以固定或者自动学习（对应不同版本的SAC算法）。

Bellman [operator](https://en.wikipedia.org/wiki/Operator_(mathematics)) $\mathcal{T}$：

$$
\mathcal{T}^\pi Q(s_t, a_t) = r(s_t, a_t) + \gamma E_{s+1 \sim p} \left[ V (s_{t+1}) \right]
$$

其中：

$$
V (s_{t+1}) = E_{a_t \sim \pi} \left[ Q (s_t, a_t) \right] - \alpha E_{a_t \sim \pi} \left[ \log \pi (a_t | s_t) \right]
$$

下一个状态的价值等于所有动作价值加上当前状态熵的期望。

分布$X$的[entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))定义为$H(X) = - E_x (\log(X))$。


### Algorithm

SAC算法是Actor-Critic结构，因而包含：

- Soft值函数：$V_\psi$
- SoftQ函数： $Q_\theta$
- 策略： $\pi_\phi$

尽管同时学习$V_\psi$和$Q_\theta$显得多余，但可以稳定训练过程。


$V_\psi$的优化目标为最小化[RSS](https://en.wikipedia.org/wiki/Residual_sum_of_squares)：

$$
J_V (\psi) = E_{s_t \sim \mathcal{D}} \left[ \frac{1}{2} \left(V_\psi (s_t) - E_{a_t \sim \pi_\phi} \left[ Q_\theta (s_t, a_t) - \log \pi_\phi (a_t | s_t) \right] \right)^2 \right]
$$

$Q_\theta$的优化目标为最小化[RSS](https://en.wikipedia.org/wiki/Residual_sum_of_squares)：

$$
J_Q (\theta) = E_{s_t \sim \mathcal{D}} \left[ \frac{1}{2} \left( Q_\theta (s_t, a_t) - r(s_t, a_t) - \gamma E_{s+1 \sim p} \left[ V (s_{t+1}) \right] \right)^2 \right]
$$

$\psi$参数更新使用了Soft更新（Polyak），$\bar{\psi}$代表更新后的网络参数。


$$
J_\pi (\phi) = E_{s_t \sim \mathcal{D}} \left[ D_{KL} \left(\pi_\phi (\cdot|s_t) \mid \mid \frac{exp( Q_{\theta} (s_t, \cdot)}{Z_\theta} \right) \right]
$$

$\frac{exp( Q_{\theta} (s_t, \cdot)}{Z_\theta}$是基于玻尔兹曼分布的探索策略，能够描述多模态策略，体现了Soft这一特点。
详见[SAC](https://zhuanlan.zhihu.com/p/70360272)。


![](/post/soft_actor_critic/projection.png)

SAC算法：


![](/post/soft_actor_critic/algorithm.png)

两个Q函数用来加速训练，损失计算中target是取二者Q中较小值。

