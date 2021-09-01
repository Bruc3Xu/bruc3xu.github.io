---
title: 'cs285 DRL notes chapter 4: Actor-Critic methods'
date: 2021-08-24 17:18:48
tags: []
published: true
hideInList: false
feature: 
isTop: false
---
回顾策略梯度算法，
$$\nabla_\theta J(\theta) \simeq \frac{1}{N}\sum_{i=1}^N\left(\sum_{t=1}^T\nabla_\theta \log\pi_\theta(a_{i,t}|s_{i,t})\right)\left(\sum_{t'=t}^T r(s_{i,t'},a_{i,t'})\right)$$
我们使用``reward-to-go'' 来近似在状态$s_{i,t}$采取动作 $a_{i,t}$的回报。上一章我们证明了这种近似具有很高的方差，这一章我们会使用其他方法来解决这个问题。

## 值函数作为基线
上一章我们为了减少方差，使用平均轨迹回报最为基线，如果我们使用平均rewrad-to-go作为基线，
$$
\bar{R_t'}=1/N\sum_{i=1}^{N}(\sum_{t'=t}^T r(s_{i,t'},a_{i,t'})
$$
而状态值函数衡量状态$s_t$的价值。定义为$V^\pi(s_t) =\sum_{t'=t}^T{\mathbb{E}_{\pi_\theta}[r(s_t',a_t')|s_t]}$，代表当前状态的期望价值，优于上式的近似表示。

$$\nabla_\theta J(\theta) \simeq \frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_{\theta}\log \pi_\theta (a_{i,t}|s_{i,t})\left(Q(s_{i,t},a_{i,t}) - V(s_{i,t})\right)$$
and the value function we used is a better approximation of the baseline $b_t = \frac{1}{N}\sum_i Q(s_{i,t},a_{i,t})$.

## 值函数拟合
我们使用监督学习来拟合值函数，损失函数
$$
L(\phi)=1/2\sum_i||\hat{V}_\phi^\pi(s_i)-y_i||^2
$$
$y_i$是当前状态的期望价值，使用bootstrap（自举）方法可以得到
$$
\begin{aligned}
y_{i,t} &= \sum_{t'=t}^{T}\mathbb{E}_{\pi_\theta}[r(s_{t'}, a_{t'})|s_{i,t}]\\
&\simeq r(s_{i,t}, a_{i,t})+\sum_{t'=t+1}^T[r(s_{t'}, a_{t'})|s_{i,t+1}]\\
&\simeq r(s_{i,t}, a_{i,t})+V^\pi(s_{i,t+1})\\
\end{aligned}
$$
尽管上式对状态价值的估计是有偏的，但具有较低的方差。
最终的MSE损失形式为
$$
L(\phi)=1/2\sum_i||\hat{V}_\phi^\pi(s_i)-(r(s_{i,t}, a_{i,t})+\hat{V}_\phi^\pi(s_{i+1}))||^2
$$

## 策略评估

