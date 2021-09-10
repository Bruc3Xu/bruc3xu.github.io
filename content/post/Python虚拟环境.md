---
title: "Python虚拟环境"
date: 2020-01-22T14:44:20+08:00
lastmod: 2020-01-22T14:44:20+08:00
draft: false
keywords: []
description: ""
tags: [python]
categories: []
author: ""

# You can also close(false) or open(true) something for this content.
# P.S. comment can only be closed
comment: false

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

## 1.多版本python
    
安装Anaconda或者Miniconda，Miniconda更加简洁，减少了许多不必要安装的内容。Conda的优势在于可以同时安装不同版本的Python，而且安装cudatools更加方便。

国内Conda的安装可以参考[Anaconda 镜像使用帮助](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)，速度更快。

创建环境`conda create -n myenv python=3.7`
    
查看所有虚拟环境`conda env list`
    
激活默认环境（base）`conda activate`或者`conda activate myenv`
    
退出当前环境`conda deactivate`

删除环境`conda remove -n envname --all`

## 2.在指定目录生成干净的Python虚拟环境
使用内置的venv，
    
生成虚拟环境`python -m venv env_dir`
    
激活虚拟环境`source ./bin/activate`
    
退出虚拟环境`deactivate`

## 3.环境迁移
### 移植conda
使用anaconda可以直接将anaconda安装目录下envs中的内容直接拷贝。
或者`conda env export > environment.yml`，然后使用`conda env create -f environment.yml`来创建新的环境。

### 使用pip

导出依赖库列表`pip freeze >requirements.txt`

下载依赖库到本地Download_File文件夹`pip download -d Download_File  -r requiremetns.txt`

在新的机器`pip install --no-index --find-links=Download_File -r requirements.txt`

### pip使用国内源
暂时使用：

`pip install numpy -i http://mirrors.aliyun.com/pypi/simple/`

长期使用：

在linux下创建`.pip/pip.conf`文件，
windows下创建`User/pip/pip.ini`文件。

文件内容
```ini
[global]
index-url = http://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host=mirrors.aliyun.com
```
