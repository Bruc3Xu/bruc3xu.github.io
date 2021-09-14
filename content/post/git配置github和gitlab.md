---
title: "Git配置github和gitlab"
date: 2020-09-20T11:11:27+08:00
lastmod: 2020-09-20T11:11:27+08:00
draft: false
keywords: []
description: ""
tags: [git]
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
在工作中，我们经常会遇到同时使用github、gitlab、自建git服务器等，下面看一下具体如何配置。
<!--more-->
这里，我们使用ssh key连接各类服务，更加安全和方便。
假设我们同时使用GitHub和gitlab。

生成github private key
  `ssh-keygen -t rsa -f ~/.ssh/id_rsa.github -C "xxx@gmail.com"`

生成gitlab private key
  `ssh-keygen -t rsa -f ~/.ssh/id_rsa.gitlab -C "xxx@foxmail.com"`

编辑ssh配置文件，使得不同域名解析正确的配置。
  `gedit ~/.ssh/config`

```config
Host github.com
  HostName github.com
  User xxx
  IdentityFile ~/.ssh/id_rsa.github
  
Host gitlab.drive.com
  HostName gitlab.com
  User xxx
  IdentityFile ~/.ssh/id_rsa.gitlab
```

最后在github和gitlab的设置中添加对应的公钥` ~/.ssh/*.pub`。
