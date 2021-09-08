---
title: "cs285 DRL notes chapter 7: Advanced Policy Gradients"
date: 2020-09-23T09:34:56+08:00
lastmod: 2020-09-23T09:34:56+08:00
draft: true
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


本章会深入策略梯度算法，进一步学习
**Natural Policy Gradient**, **Trust Region Policy Optimization**, or **Proximal Policy
Optimization**等算法。

## Policy Gradient as Policy Iteration
回顾Policy Gradient和Actor-Critic算法，
$$
\nabla_{\theta}J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T
\nabla_{\theta}\log \pi_{\theta}(a_{i, t} \vert s_{i, t}) A^{\pi}_{i, t}
$$

可以看作一下过程
1. 使用当前策略$\pi$估计$A^{\pi}_{i, t}$
2. 基于$A^{\pi}(s_t, a_t)$来获得改进后的策略$\pi'$

与Policy Iteration过程相同。 什么情况下Policy Gradient可以看作Policy Iteration？首先我们分析Policy Gradient中的**policy improvement**。

策略梯度：

$$
J(\theta) = E_{\tau \sim p_{\theta}(\tau)}\left[ \sum_{t}\gamma^t r(s_t, a_t) \right]
$$

新策略参数$\theta'$和旧策略参数$\theta$ 的policy improvement：
$$
J(\theta') - J(\theta) = E_{\tau \sim p_{\theta'}(\tau)} \left[
\sum_t \gamma^t A^{\pi_{\theta}}(s_t, a_t) \right]
$$
旧策略$\theta$的advantage关于新策略$\theta'$的trajectory的期望值。

That is the expected total advantage with respect to the parameters  **under the
distribution induced by the new parameters **. This is very important,
because this improvement objective is the same of
[Policy Iteration](/lectures/lecture7). If we can show that the gradient of this improvement
is the same gradient of Policy Gradient, then we can show that Policy Gradient moves in the
direction of improving the same thing as Policy Iteration.

如果按照policy iteration的流程，在improvement中，也就只需要使得每步提升最大，找到新的parameter使得等式的右方最大化即可。但是improvement是计算新策略$\theta'$,
的轨迹期望，而我们当前的样本都是基于旧策略$\theta$采集的。

把improvement展开：

$$
E_{\tau \sim p_{\theta'}(\tau)} \left[ \sum_t \gamma^t A^{\pi_{\theta}}(s_t, a_t) \right] =
\sum_t E_{s_{t} \sim p_{\theta'}(s_{t})} \left[ E_{a_{t} \sim \pi_{\theta'}(a_{t})} \left[
\gamma^t A^{\pi_{\theta}}(s_t, a_t)
\right]\right]
$$

然后使用[Importance Sampling](https://en.wikipedia.org/wiki/Importance_sampling)将动作从分布$\pi_{\theta'}(a_t)$转换到分布$\pi_{\theta}(a_t)$:
$$
E_{\tau \sim p_{\theta'}(\tau)} \left[ \sum_t \gamma^t A^{\pi_{\theta}}(s_t, a_t) \right] =
\sum_t E_{s_{t} \sim p_{\theta'}(s_{t})} \left[ E_{a_{t} \sim \pi_{\theta}(a_{t})} \left[
\frac{\pi_{\theta^{\prime}}(a_t)}{\pi_{\theta}(a_t)} \gamma^t A^{\pi_{\theta}}(s_t, a_t)
\right]\right]
$$

但是现在状态$s_t$仍然是来自于分布$p_{\theta'}$。
如果$\pi$和$\pi'$的**total variation divergence**（总变异散度，两个分布每个元素差值绝对值之和）小于$\epsilon$，

$$
\vert \pi_{\theta'}(a_t \vert s_t) - \pi_{\theta}(a_t \vert s_t) \vert \le \epsilon
$$

那么$p_{\theta'}$和$p_{\theta}$的总变异散度有上界：

$$
\vert p_{\theta'}(s_t) - p_{\theta}(s_t) \vert \le 2\epsilon t
$$

因而，improvement有上界

$$
J(\theta') - J(\theta) \le \sum_{t} 2\epsilon t C
$$

其中$r_{max}$是最大单步奖励，$C \in O(\frac{r_{max}}{1 - 
\gamma})$。

最终，我们用$p_{\theta}$来替换
$p_{\theta'}$ 

$$
\begin{aligned}
\overline{A}(\theta') &= \sum_t E_{s_t \sim p_{\theta}(s_t)}\left[
E_{a_t \sim \pi_{\theta}(a_t \vert s_t)}\left[
\frac{\pi_{\theta'}(a_t \vert s_t)}{\pi_{\theta}(a_t \vert s_t)} \gamma^t
A^{\pi_{\theta}}(s_t, a_t) \right] \right]\\\\
\theta' &\leftarrow \arg\max_{\theta'} \overline{A}(\theta')   
\end{aligned}
$$

### A better measure of divergence
在现实中，我们很难计算两个分布的total variation divergence，一般使用
[KL Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)替代。

实际上，两个分布$\pi_{\theta'}(a_t \vert s_t)$和
$\pi_{\theta}(a_t \vert s_t)$的total variation divergence上界可以用KL divergence($D_{KL}$)的均方根来表示。

$$
\vert \pi_{\theta'}(a_t \vert s_t) - \pi_{\theta}(a_t \vert s_t) \vert \le
\sqrt{\frac{1}{2} D_{KL}\left(\pi_{\theta'}(a_t \vert s_t) \vert\vert 
\pi_{\theta}(a_t \vert s_t)\right)}
$$


## Enforcing the KL constraint
上文所有的证明都建立在$\pi'$和$\pi$之间的差异不大的基础上，因而需要有一个限制：

$$
D_{KL}\left(\pi_{\theta'(a_t \vert s_t)} \vert \vert \pi_{\theta}(a_t \vert s_t)\right)
\le \epsilon
$$

### Dual Gradient Descent
We can subtract the KL Divergence to the maximization objective of Eq. \ref{eq:opt_objective}

$$
\label{eq:dual_opt}
\mathcal{L}(\theta', \lambda) = \overline{A}(\theta') - \lambda \left(D_{KL}\left(\pi_{\theta'}
(a_t \vert s_t) \vert\vert \pi_{\theta}(a_t \vert s_t)\right) - \epsilon\right)
$$

and perform a **dual gradient descent** by repeating the following steps:
1. Maximize $\mathcal{L}(\theta', \lambda)$ with respect to $\theta'$ (usually not until
   convergence, but only a few maximization steps)
2. Update $\lambda \leftarrow \lambda + \alpha \left( D_{KL}\left(\pi_{\theta'}(a_t \vert s_t)
   \vert\vert \pi_{\theta}(a_t \vert s_t)\right) - \epsilon \right)$

### Natural Gradients
One way to optimize a function within a certain range is, provided that the range is small
enough, to take a first order Taylor expansion and optimize it instead. If we take a linear
expansion of our objective and we evaluate it at $\theta$, we obtain the usual Policy Gradient:

$$
\nabla_{\theta}\overline{A}(\theta) = 
\sum_t E_{s_t \sim p_{\theta}(s_t)}\left[ E_{a_t \sim p_{\theta}(a_t \vert s_t)} \left[
\gamma^t \nabla_{\theta}\log\pi_{\theta}(a_t \vert s_t) A^{\pi_{\theta}}(s_t, a_t)
\right]\right]
$$
since the importance sampling ratio in $\theta$ becomes 1. See 
[Lecture 5: Policy Gradients](/lectures/lecture5) for the gradient derivation.

Taking a Policy Gradient step means that we are taking a step in a circular radius around
$\theta$, which is equivalent of maximizing $\overline{A}$ subject to
$$
\label{eq:circle_constr}
\vert\vert \theta' - \theta \vert\vert^2 \le \epsilon
$$
However, our constraint is that of Eq. \ref{eq:kl_constraint}. We therefore take a second order
approximation of the KL Divergence

$$
D_{KL}(\pi_{\theta'} \vert\vert \pi_{\theta}) \approx \frac{1}{2} (\theta' - \theta) \pmb{F}
(\theta' - \theta)
$$
where $\pmb{F}$ is the [Fischer Information Matrix](https://en.wikipedia.org/wiki/Fisher_information_metric#Relation_to_the_Kullback%E2%80%93Leibler_divergence). This makes our optimization
region becoming an ellipse inside which the KL constraint is respected.

![](/post/cs285_chapter7/gradients.png)

The figure above shows the optimization regions given, respectively, by the "naive" constraint
of Eq. \ref{eq:circle_constr} implied by the and the usual Policy Gradient algorithm, and that
given by the $D_{KL}$ constraint of Eq. \ref{eq:kl_constraint}.
 
We transform our objective by $\pmb{F}$ inverse, such that the optimiation region becomes
again a circle and we can take a gradient step on this transformation. We call this the
**Natural Gradient**:

$$
\theta' = \theta + \alpha \pmb{F}^{-1} \nabla_{\theta}J(\theta)
$$
The learning rate $\alpha$ must be chosen carefully. More on Natural Gradients in
[J. Peters, S. Schaal, Reinforcement learning of motor skills with policy gradients](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.142.8735&rep=rep1&type=pdf)

The use of a natural gradient however does not come without its issues. The Fischer Information
Matrix is defined as
$$
\pmb{F} = E_{\pi_{\theta}}\left[
\nabla_{\theta}\log\pi_{\theta}(\pmb{a}\vert\pmb{s})
\nabla_{\theta}\log\pi_{\theta}(\pmb{a}\vert\pmb{s})^T
\right]
$$
which is the outer product of the gradient logs. Therefore, if $\theta$ has a million
parameters, $\pmb{F}$ will be a million by a million matrix, and computing its inverse would
become infeasible. Moreover, since it is an expectation, we also need to compute it from samples,
which again increases the computational cost.

### Trust Region Policy Optimization
While in the paragraph above we chose the learning rate ourselves, we may want to instead choose
$\epsilon$ and enforce each gradient step to be ecactly $\epsilon$ in $D_{KL}$ variation.
We therefore use a learning rate $\alpha$ according to
$$
\alpha = \sqrt{\frac{2\epsilon}{\nabla_{\theta}J(\theta)^T\pmb{F}\nabla_{\theta}J(\theta)}}
$$
[Schulman et al., Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf)
introduced the homonymous algorithm, and, most importantly, provides a **efficient way of
computing the matrix** $\pmb{F}$.


### Proximal Policy Optimization
[Schulman et al., Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347.pdf),
proposes a way of enforcing the $D_{KL}$ constraint without the need of computing the
Fischer Information Matrix or its approximation. This can obtained in two ways:

#### Clipping the surrogate objective
Let $r(\theta)$ be the Importance Sampling ratio of the Eq. \ref{eq:opt_objective} objective.
Here, we maximize instead a clipped objective
$$
L^{CLIP} = E_t \left[ \min\left(
r_r(\theta)A_t,
clip(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t
\right) \right]
$$ 

The following figure shows a single term in $L^{CLIP}$ for positive and negative
advantage.

![](/post/cs285_chapter7/ppo_objective.png)

However, recent papers such as [Engstrom et al., Implementation Matters in Deep Policy Gradients](https://openreview.net/pdf?id=r1etN1rtPB)
show how this clipping mechanism does not prevent the gradient steps to violate the KL
constraint. Furthermore, they claim that the effectiveness that made PPO famous comes from its
**code-level optimizations**, and TRPO above may actually be better if these are implemented.

#### Adaptive KL Penalty Coefficient
Another approach described by the PPO paper is similar to the dual gradient descent we described
above. It consists in repeating the following steps in each policy update:
- Using several epochs of minibatch SGD, optimize the KL-penalized objective

$
L^{KLPEN}(\theta) = E_t \left[ \frac{\pi_{\theta'}(a_t \vert s_t)}{\pi_{\theta}(a_t \vert s_t)}
A^{\pi_{\theta}}(s_t, a_t) - \beta D_{KL}(\pi_{\theta'}(. \vert s_t)\vert\vert
\pi_{\theta}(. \vert s_t))
\right]
$

- Compute $d = E_t \left[ D_{KL}(\pi_{\theta'}(. \vert s_t)\vert\vert 
\pi_{\theta}(. \vert s_t)) \right] $
    - If $d \lt d_{targ}/1.5$ then $\beta \leftarrow \beta/2$
    - If $d \gt 1.5 d_{targ}$ then $\beta \leftarrow 2\beta$

Where $d_{targ}$ is the desired KL Divergence target value.
