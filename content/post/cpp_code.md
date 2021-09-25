---
title: "Cpp编程规范"
date: 2021-05-23T09:20:09+08:00
lastmod: 2021-05-23T09:20:09+08:00
draft: false
keywords: []
description: ""
tags: []
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
# 代码风格检查
## 1.1. 代码规范
尽量遵循 Google C++ Style Guide
<!--more-->
 [[**中文版**](https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/) ] [[**英文版**](https://google.github.io/styleguide/cppguide.html)]。
## 1.2. 工具安装与配置
    
  使用 **[** [**cpplint**](https://github.com/cpplint/cpplint) **]** 工具进行检查，cpplint是Google开发的一个C++风格检测工具，检测代码是否符合Google C++ Style。
### 1.2.1. cpplint安装
    
  (1) 安装python和包管理工具
    
  (2) 安装cpplint，如果通过pip安装，输入命令：`sudo pip3 install cpplint`

### 1.2.2. VS Code 配置 cpplint
  VS Code作为一款轻量但是功能非常强大的代码编辑器，可以配置cpplint插件进行风格检查。
    
  (1) 在 **Extensions Tab** 中，搜索并安装cpplint插件
    
  (2) 打开 **Extensions Settings** ，配置cpplint路径等设置
    
  (3) 每次打开C++文件，插件便会自动调用cpplint工具检测代码风格，并将检测结果输出到 **PROBLEMS** 窗口中，双击会定位到具体代码行
    
  (4) 也可以通过 **F1** 打开命令面板，然后调用cpplint工具
### 1.2.3. 常见报错类型
    
  (1) 每行长度不得80字符
    
  (2) 使用更安全的函数
    
  (3) 注意代码缩进，用空格代替Tab
    
  (4) "{"应在上一行结尾，不应另起一行
    
  (5) "else"应在上一行的"}"后面，不应另起一行
    
  (6) 使用逗号应与前后字符间距一个空格
    
  (7) 在代码后添加注释时，"//"应与前面的代码间隔两个空格，与后面的注释间隔一个空格
    
  (8) 删去代码行末尾的空格以及代码间多余的空行

# 2. 静态检查
## 2.1. Cppcheck检查类型
    
  Cppcheck是一种C/C++代码缺陷检查工具，不检查语法错误，只检查编译器检查不出来的bug类型，可以作为编译器的一种补充检查。Cppcheck执行的检查包括：
    
  (1) 空指针检查
    
  (2) 数据越界
    
  (3) 内存泄漏
    
  (4) 野指针
    
  (5) 逻辑错误，算法函数形参能力集判断，重复的代码分支，bool类型和INT进行比较，表达式永远True或者false等共18类检查
    
  (6) 可疑代码检查，if判断中含有可疑的=号，自由变量返回局部变量等共计15类检查
    
  (7) 运算错误，判断无符号数小于0,对bool类型进行++自增，被除数非零判断等，共计11类检查
## 2.2. Cppcheck安装与配置
### 2.2.1. Windows平台
    
  (1) 下载并安装 **[** [**Cppcheck Win64安装包**](https://github.com/danmar/cppcheck/releases/download/2.4.1/cppcheck-2.4.1-x64-Setup.msi) **]**
### 2.2.2. Linux平台
    
  (1) Debian平台：`sudo apt-get install cppcheck`
    
  (2) Fedora平台：`sudo yum install cppcheck`
### 2.2.3. 常用IDE插件
    
  按需 **[** [**Cppcheck官网**](http://cppcheck.sourceforge.net/) **]** 下载安装即可
## 2.3. Cppcheck使用（#待补充）
    
  支持Command Line模式、GUI模式、IDE插件模式，具体参考 **[** [**Cppcheck手册**](http://cppcheck.sourceforge.net/manual.pdf) **]**
# 3. 动态检查
## 3.1. 内存泄漏检查
### 3.1.1. Windows平台（#待补充）
    
  **[** [**Visual Leak Detector工具**](https://kinddragon.github.io/vld/) **]**
### 3.1.2. Linux平台（#待补充）
    
  **[** [**Valgrind Memcheck工具**](https://valgrind.org/) **]**
