---
title: "Red Black Tree"
date: 2021-09-01T17:17:29+08:00
lastmod: 2021-09-01T17:17:29+08:00
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
# 红黑树（Red-black tree）
[reference](https://zhuanlan.zhihu.com/p/273829162)

红黑树是一种自平衡二叉查找树。可以在$O(\log n)$时间内完成查找、插入和删除，这里的n是树中元素的数目。

2-3-4树模型：存在2节点、3节点、4节点的B树（平衡树）。
- 2节点：节点有一个key，两个指针分别指向大于key和小于key的节点。
- 3节点：两个key，3个指针指向不同区间的节点。
- 4节点：3个key，4个指针指向不同区间的节点。

红黑树是2-3-4模型的一种实现，为了不进行不同节点的转换，选择在二叉树的基础上加入颜色属性来表示2-3-4模型中不同类型的节点。

2节点为黑色节点，3、4节点表示为红色节点和黑色节点的组合。