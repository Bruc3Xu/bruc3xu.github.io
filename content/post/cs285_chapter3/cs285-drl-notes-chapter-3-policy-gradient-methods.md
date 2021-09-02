---
title: 'cs285 DRL notes chapter 3: policy gradient methods'
date: 2020-08-24 16:00:06
tags: [reinforcement learning,cs285]
published: true
hideInList: false
isTop: false
---
回顾强化学习的目标，我们希望获得策略的最优参数$\theta^*$，

$$
\theta^*=\underset{\theta}{argmax}\mathbb{E}\_{\tau\sim p_{\theta}(\tau)}[\sum_{t=1}^{t=T}r(s_t, a_t)]
$$

这实际上是一个优化问题，因此我们可以使用多种优化方法来优化这个目标，例如梯度下降。我们将优化的目标函数定义为$J(\theta)$：
$$
J(\theta) = \mathbb{E}\_{\tau\sim \pi_\theta(\tau)}[r(\tau)]
$$
其中，$r(\tau)$是轨迹累积奖励, 等价于$$\sum_{t=1}^{T} r(s_t,a_t)$$。$J$进一步写作：

$$
J(\theta) = \int\pi_\theta r(\tau)d\tau
$$

## 策略梯度定理(Policy Gradient Theorem)
一个小技巧，
$$
\pi_\theta(\tau)\nabla_\theta \log \pi_\theta(\tau) = \pi_\theta(\tau)\frac{\nabla_\theta \pi_\theta(\tau)}{\pi_\theta(\tau)}=\nabla_\theta \pi_\theta (\tau)
$$

利用上式，$J(\theta)$的梯度可以表示为

$$
\begin{aligned} \nabla_\theta J(\theta) &= \int \nabla_\theta \pi_\theta r(\tau)d\tau \\\\
&= \int\pi_\theta(\tau)\nabla_\theta \log \pi_\theta(\tau) r(\tau) d\tau \\\\
&= \mathbb{E}\_{\tau\sim \pi_\theta(\tau)}[\nabla_\theta \log \pi_\theta (\tau)r(\tau)] 
\end{aligned}
$$


轨迹$\tau$是状态、动作的时间序列，因此由$\pi_\theta$得到的轨迹概率，根据贝叶斯定理为$$\pi_\theta(s_1,a_1,...,s_T,a_T) = p(s_1)\prod_{t=1}^T\pi_\theta (a_t|s_t)p(s_{t+1}|s_t,a_t)$$。

两边取对数， $$\log\pi_\theta(\tau) = \log p(s_1) + \sum_{t=1}^T \log\pi_\theta(a_t|s_t)+\log p(s_{t+1}|s_t,a_t)$$

将其带入策略梯度计算中，

$$
\begin{aligned} \nabla_\theta J(\theta) &= \mathbb{E}_{\tau\sim \pi_\theta(\tau)}\left[\nabla_\theta\left(\log p(s_1) + \sum_{t=1}^T \log \pi_\theta(a_t|s_t)+\log p(s_{t+1}|s_t,a_t)\right)r(\tau)\right] \\\\
&= \mathbb{E}_{\tau\sim \pi_\theta(\tau)}\left[ \left(\sum_{t=1}^T\nabla_\theta \log\pi_\theta(a_t|s_t)\right)\left(\sum_{t=1}^T r(s_t,a_t)\right)\right]
\end{aligned}
$$

## 策略梯度评估
从上式可以看出，策略梯度计算是轨迹$\tau$的梯度的期望，然而轨迹空间极大，难以计算。这时，我们使用近似的方法来求解，具体地可以使用Monte-Carlo近似，即计算N个样本的平均值作为策略梯度的近似。

$$\nabla_\theta J(\theta) \simeq \frac{1}{N}\sum_{i=1}^N\left(\sum_{t=1}^T\nabla_\theta \log\pi_\theta(a_{i,t}|s_{i,t})\right)\left(\sum_{t=1}^T r(s_{i,t},a_{i,t})\right)$$

其中$i, t$分别代表轨迹$i$和时间步$t$。

可以看出，上式中并没有出现转移概率函数，即没有用到马尔可夫性质，因此适用于POMDP。

基于上式，我们可以使用梯度上升算法来优化参数$\theta$
$\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta)$

此类依赖蒙特卡洛近似方法的策略梯度算法（vanilla policy gradient）称为ReinForce算法。
***
Base policy $\pi_\theta(a_t|s_t)$, sample trajectories $\tau^i$\
WHILE True\
$\quad$ Sample ${\tau^i}$ from $\pi_\theta(a_t|s_t)$\
$\quad \nabla_\theta J(\theta) \simeq \frac{1}{N}\sum_i\left(\sum_t\nabla_\theta \log\pi_\theta(a_{i,t}|s_{i,t})\right)\left(\sum_t r(s_{i,t},a_{i,t})\right)$ \
$\quad$ Improve policy by $\theta \leftarrow \theta + \alpha\nabla_\theta J(\theta)$\
Return optimal trajectory from gradient ascent as $\tau^{return}$
***

## 策略梯度背后
策略的最大似然定义为

$$\nabla_\theta J(\theta) \simeq \frac{1}{N}\sum_{i=1}^N\nabla_\theta \log\pi_\theta(\tau_i)$$

而策略梯度为

$$\nabla_\theta J(\theta) \simeq \frac{1}{N}\sum_{i=1}^N\nabla_\theta \log\pi_\theta(\tau_i)r(\tau_i)$$

直观上来说，我们不断优化策略，分配回报高的轨迹更大的权重，因而发生的概率更大。

## 策略梯度的高方差
1）对于两条轨迹，轨迹回报分别为1和-1，两条轨迹的概率会相应增加和减少。但如果为两条轨迹同时加上一个常数，并不会影响轨迹的奖励分布，但两条轨迹的概率都增大了。这体现了策略梯度算法的高方差。
2）假设环境具有正的回报，对于采样的动作，其发生概率会增大，间接减少了其他动作的发生概率，而这些动作可能是好的动作。
## 减少策略梯度算法的方差
### 因果关系
实际上，智能体更关心做出动作后的回报，而不是轨迹的回报，因为之前的奖励和当前动作没有因果关系。我们定义在当前时刻t之后的回报为reward-to-go，

$$
R_t=\sum_{t=t'}^{T}r(s_t, a_t, s_{t+1})
$$

策略梯度为

$$\nabla_\theta J(\theta) \simeq \frac{1}{N}\sum_{i=1}^N\left(\sum_{t=1}^T\nabla_\theta \log\pi_\theta(a_{i,t}|s_{i,t})\right)\left(\sum_{t'=t}^T r(s_{i,t'},a_{i,t'}, s_{i,t'+1})\right)$$

方差由于奖励累加项的减少而减小。
### 基线
我们并不会让所有回报大于0的动作都去增大它的概率，而是设立一个基线，对回报大于基线的动作才增大它的概率。自然地，可以将平均回报最为基线。

$$
b=1/N\sum_{i=1}^{N}r(\tau)
$$

策略梯度为
$$\nabla_\theta J(\theta) \simeq \frac{1}{N}\sum_{i=1}^N\nabla_\theta \log\pi_\theta(\tau_i)[r(\tau_i)-b]$$

由于

$$
\begin{aligned}
\mathbb{E}\_{\pi_\theta(\tau)}[\nabla_\theta\log \pi_\theta(\tau)b]&=\int \pi_\theta(\tau)\nabla \log\pi_\theta(\tau)bd\tau\\\\
&=\int \nabla_\theta \pi_\theta(\tau)bd\tau\\\\
&=b\nabla_\theta\int\pi_\theta(\tau)d\tau\\\\
&=b\nabla_\theta 1\\\\
&=0
\end{aligned}
$$

因此，加上baseline之后，策略梯度仍然是无偏的。
### 方差分析
方差定义为
$$\mathrm{Var}[x] = \mathbb{E}[x^2]-\mathbb{E}[x]^2$$

具有基线的策略梯度为
$$\nabla_\theta J(\theta) \simeq \mathbb{E}_{\tau\sim\pi_\theta(\tau)}\left[\nabla_\theta \log\pi_\theta(\tau)\left(r(\tau)-b\right)\right]$$

因此，方差为
$$\mathrm{Var} = \mathbb{E}\_{\tau\sim\pi_\theta(\tau)}\left[\left(\nabla_\theta \log\pi_\theta(\tau)\left(r(\tau)-b\right)\right)^2\right] - \mathbb{E}_{\tau\sim\pi_\theta(\tau)}\left[\nabla_\theta \log\pi_\theta(\tau)\left(r(\tau)-b\right) \right]^2$$

因为加上基线也是无偏的，上式第二项可以写作
 $$\mathbb{E}\_{\tau\sim\pi_\theta(\tau)}\left[\nabla_\theta \log\pi_\theta(\tau)r(\tau) \right]^2$$

这里我们计算方差关于b的导数，以求解最优的b

$$
\begin{aligned}
\frac{d\mathrm{Var}}{db} &= \frac{d}{db}\mathbb{E}\left[g(\tau)^2(r(\tau)-b)^2\right]\\\\ 
&=\frac{d}{db}\mathbb{E}\left[g(\tau)^2r(\tau)^2\right] - 2\mathbb{E}\left[g(\tau)^2r(\tau)b\right] + b^2\mathbb{E}\left[g(\tau)^2\right]\\\\
&=-2\mathbb{E}\left[ g(\tau)^2r(\tau)\right]+2b\mathbb{E}\left[g(\tau)^2\right]\\\\
&=0
\end{aligned}
$$

得到

$$b^{opt} = \frac{\mathbb{E}\left[g(\tau)^2r(\tau)\right]}{\mathbb{E}\left[g(\tau)^2\right]}$$

## 伪代码实现
```python
traj = policy.explore()
logits = policy.predict(traj.states)
negative_likelihood = torch.SoftMax(logits).gather(1, traj.actions)
loss = (traj.q_vals * negative_likelihood).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## on-policy vs off-policy
on-policy指同策略，只从当前策略采样得到额数据中学习。off-policy指异策略，不仅从当前策略，还从其他策略采样得到的数据学习。策略梯度算法是同策略的学习方法，每次更新策略后，旧的样本就要丢弃，无疑是低效的。这里，我们可以使用异策略学习方法。
### 重要性采样（Importance Sampling）
给定分布$p(x)$，如何计算从分布$q(x)$得到的样本的期望? 重要性采样的思想是使用重要性权重，计算另一个分布的期望，因而能够进行异策略的学习。

重要性采样中

$$
\begin{aligned}
\mathbb{E}\_{x\sim p(x)}\left[f(x)\right] &= \int p(x)f(x)\;dx\\\\
&=\int \frac{q(x)}{q(x)}p(x)f(x)\;dx\\\\
&=\mathbb{E}_{x\sim q(x)}\left[\frac{p(x)}{q(x)}f(x)\right]
\end{aligned}
$$

假设学习的策略是$\pi_\theta(\tau)$，行为策略是 $\bar{\pi}(\tau)$，使用从$\bar{\pi}(\tau)$采样得到的样本来计算$J(\theta)$

$$
\begin{aligned}
J(\theta) &= \mathbb{E}\_{\tau\sim\pi_\theta(\tau)}\left[r(\tau)\right]\\\\
&= \mathbb{E}_{\tau\sim\bar{\pi}(\tau)}\left[ \frac{\pi_\theta(\tau)}{\bar{\pi}(\tau)} r(\tau)\right]
\end{aligned}
$$

回顾

$$\pi_\theta(\tau) = p(s_1)\prod_{t=1}^T\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)$$. 

得到

$$
\begin{aligned}
\frac{\pi_\theta(\tau)}{\bar{\pi}(\tau)} &=\frac{p(s_1)\prod_{t=1}^T\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)}{p(s_1)\prod_{t=1}^T\bar{\pi}(a_t|s_t)p(s_{t+1}|s_t,a_t)}\\
&= \frac{\prod_{t=1}^T\pi_\theta(a_t|s_t)}{\prod_{t=1}^T\bar{\pi}(a_t|s_t)}
\end{aligned}
$$

如果我们想要从旧的策略$\pi_\theta$采样数据中学习新的参数$\theta'$，使用重要性采样

$$J(\theta') = \mathbb{E}\_{\tau\sim\pi_\theta(\tau)}\left[\frac{\pi_{\theta'}(\tau)}{\pi_\theta(\tau)}r(\tau)\right]$$

策略梯度为：

$$
\begin{aligned}
\nabla_{\theta'}J(\theta')&=\mathbb{E}_{\tau\sim\pi_\theta(\tau)}\left[\frac{\pi_{\theta'}(\tau)}{\pi_\theta(\tau)}\nabla_{\theta'}\log\pi_{\theta'}(\tau)r(\tau)\right]\\\\
&= \mathbb{E}_{\tau\sim\pi_\theta(\tau)}\left[  \left(  \frac{\prod_{t=1}^T\pi_{\theta'}(a_t|s_t)}{\prod_{t=1}^T\pi_{\theta}(a_t|s_t)} \right)\left( \sum_{t=1}^T\nabla_{\theta'}\log\pi_{\theta'}(a_t|s_t)  \right)\left(\sum_{t=1}^Tr(s_t,a_t)\right)  \right]
\end{aligned}
$$

当T相当大时，连乘的结果可能极大或极小，增大方差。参考上文reward-to-go，将来的动作并不影响现在的重要性权重，因此可以将其截断。

$$
\nabla_{\theta'}J(\theta')=\mathbb{E}_{\tau\sim\pi_\theta(\tau)}\left[\sum_{t=1}^T\nabla_{\theta'}\log\pi_{\theta'}(a_t|s_t) \left(\prod_{t'=1}^t \frac{\pi_{\theta'}(a_{t'}|s_{t'}) }{\pi_{\theta}(a_{t'}|s_{t'})}\right)    \left(  \sum_{t'=t}^Tr(s_{t'},a_{t'}))    \right)\right]
$$

## TRPO
TODO
## PPO
TODO