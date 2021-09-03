---
title: "Q_prop"
date: 2021-09-03T11:23:18+08:00
lastmod: 2021-09-03T11:23:18+08:00
draft: true
keywords: []
description: ""
tags: []
categories: []
author: ""

# You can also close(false) or open(true) something for this content.
# P.S. comment can only be closed
comment: false
toc: false
autoCollapseToc: false
postMetaInFooter: false
hiddenFromHomePage: false
# You can also define another contentCopyright. e.g. contentCopyright: "This is another copyright."
contentCopyright: false
reward: false
mathjax: false
mathjaxEnableSingleDollar: false
mathjaxEnableAutoNumber: false

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
## control variates
控制变量法（英语：control variates）是在蒙特卡洛方法中用于减少方差的一种技术方法。该方法通过对已知量的了解来减少对未知量估计的误差。

假设要估计的参数为$\mu$。同时对于统计m，其期望值为$\mu ：\mathbb {E} \left[m\right]=\mu$，即m是$\mu$的无偏差估计。此时，对于另一个统计t，已知$\mathbb {E} \left[t\right]=\tau$。于是，
$m^{\star }=m+c\left(t-\tau \right)$
也是$\mu$的无偏差估计，c为任一给定系数。$m^{\star}$的方差为

$$
\textrm{Var}\left(m^{\star}\right)=\textrm{Var}\left(m\right)+c^2\textrm {Var}\left(t\right)+2c\textrm{Cov}\left(m,t\right);
$$

可以证明，使得方差最小的系数c为

$$
c^{\star }=-{\frac {{\textrm {Cov}}\left(m,t\right)}{{\textrm {Var}}\left(t\right)}};
$$

此时，对应的方差则为

$$
\begin{aligned}
{\textrm {Var}}\left(m^{\star }\right)&={\textrm {Var}}\left(m\right)-{\frac {\left[{\textrm {Cov}}\left(m,t\right)\right]^{2}}{{\textrm {Var}}\left(t\right)}}\\\\
&=\left(1-\rho _{m,t}^{2}\right){\textrm {Var}}\left(m\right);
\end{aligned}
$$

其中
$\rho_{m,t}={\textrm {Corr}}\left(m,t\right)\,$
为m与t之间的相关系数。$\rho_{m,t}$越大时，方差越小。

当$\textrm{Cov}\left(m,t\right),\textrm{Var}\left(t\right)或\rho_{m,t}$未知时，可以通过蒙特卡洛模拟进行估计。由于该方法相当于一个最小二乘法系统，又被称为回归抽样（regression sampling）。
## q-prop