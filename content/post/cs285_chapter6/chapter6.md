---
title: "cs285 DRL notes chapter 6: Deep RL with Q-Functions"
date: 2020-09-18T15:46:38+08:00
lastmod: 2020-09-18T15:46:38+08:00
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
# Deep Q Network
## 在线Q-Learning的问题
我们使用神经网路来近似Q函数，而神经网络学习数据一般要求独立同分布（i.i.d），但是在rl的采样中，序列决策问题的数据来源于与环境的相邻交互，数据之间往往是具有相关性的。即可能状态相似，但奖励相差很多，训练

## replay buffers


## target network


## general view of DQN