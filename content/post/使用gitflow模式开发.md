---
title: "使用gitflow模式开发"
date: 2021-04-24T11:19:17+08:00
lastmod: 2021-04-24T11:19:17+08:00
draft: false
keywords: []
description: ""
tags: [git]
categories: []
author: ""

# You can also close(false) or open(true) something for this content.
# P.S. comment can only be closed
comment: false
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
Gitflow工作流通过为功能开发、发布准备和维护分配独立的分支，让发布迭代过程更流畅，非常适合用来管理大型项目的发布和维护
<!--more-->
首先，看图
![](/post/git-flow.png)

## 不同分支解释
- 1.初始分支：master分支commit都应tag
- 2.分支名 feature/*
    
  Feature分支做完后，必须合并回Develop分支, 合并完分支后一般会删点这个Feature分支，但是我们也可以保留
- 3.分支名 release/*
    
  Release分支 **基于Develop分支创建** ，打完Release分之后，我们可以在这个 **Release分支上测试，修改Bug** 等。同时，其它开发人员可以基于开发新的Feature (记住：一旦打了Release分支之后不要从Develop分支上合并新的改动到Release分支)
    
  **发布Release分支时，合并Release到Master和Develop** ， 同时在 **Master分支上打个Tag记住Release版本** 号，然后可以删除Release分支了。
- 4.分支名 hotfix/*
    
  hotfix分支基于Master分支创建， **开发完后需要合并回Master和Develop分支** ，同时在Master上打一个tag
## 命令
    
  a.创建develop分支：
```bash
  git branch develop
    
  git push -u origin develop
```
  b.开始新的feature
```bash 
  git checkout -b feature/* develop
    
  git push -u origin feature/*
    
  git status
    
  git add file
    
  git commit
```
  c.完成feature
```bash
  git pull origin develop
    
  git checkout develop
    
  git merge –no-ff feature/*
    
  git push origin develop
    
  git branch -d some feature
    
  git push origin –delete feature/*
```
  d.开始release
```bash
  git checkout -b release-0.10.0 develop
```
  3.完成release
```bash

git checkout master
git merge --no-ff release-0.1.0
git push
  
git checkout develop
git merge --no-ff release-0.1.0
git push
  
git branch -d release-0.1.0
  
# If you pushed branch to origin:
git push origin --delete release-0.1.0   
  
git tag -a v0.1.0 master
git push --tags
  
  ```
    
  4.开始Hotfix
    
  ``` bash
git checkout -b hotfix-0.1.1 master    
  
#完成Hotfix
  
git checkout master git merge –no-ff hotfix-0.1.1 git push
  
git checkout develop git merge –no-ff hotfix-0.1.1 git pushgit branch -d hotfix-0.1.1
git tag -a v0.1.1 master git push –tags 
  ```
## 使用git flow script
如果偏向于使用图形界面。可以使用git Kraken
### 安装
- OS X
`brew install git-flow`
- Linux
    
  `apt-get install git-flow`
- Windows
    
  wget -q -O - –no-check-certificate [https://github.com/nvie/gitflow/raw/develop/contrib/gitflow-installer.sh](https://github.com/nvie/gitflow/raw/develop/contrib/gitflow-installer.sh) | bash
### 使用
- **初始化:** git flow init
- **开始新Feature:** git flow feature start MYFEATURE
- **Publish一个Feature(也就是push到远程):** git flow feature publish MYFEATURE
- **获取Publish的Feature:** git flow feature pull origin MYFEATURE
- **完成一个Feature:** git flow feature finish MYFEATURE
- **开始一个Release:** git flow release start RELEASE [BASE]
- **Publish一个Release:** git flow release publish RELEASE
- **发布Release:** git flow release finish RELEASE 别忘了git push –tags
- **开始一个Hotfix:** git flow hotfix start VERSION [BASENAME]
- **发布一个Hotfix:** git flow hotfix finish VERSION

