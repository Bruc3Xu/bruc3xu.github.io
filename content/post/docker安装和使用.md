---
title: "Docker安装和使用"
date: 2020-09-14T10:15:36+08:00
lastmod: 2020-09-14T10:15:36+08:00
draft: false
keywords: [docker]
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
本文介绍了docker、nvidia-docker安装以及多阶段打包镜像的过程。
<!--more-->
# 安装docker
## step 1: 安装必要的一些系统工具
  ``` bash
sudo apt-get update 
sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common
  ```
## step 2: 安装GPG证书
  ``` bash
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key add -
  ```
## Step 3: 写入软件源信息 
  `sudo add-apt-repository "deb [arch=amd64] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"`

如果是基于ubuntu的衍生版本，改成ubuntu版本
## Step 4: 更新并安装Docker-CE 
  `sudo apt update && sudo apt-get -y install docker-ce`
## 安装指定版本的Docker-CE:
  `sudo apt-get -y install docker-ce=[VERSION]`
## 添加docker组 
  `sudo usermod -aG docker your-user`
## 配置镜像加速器 
  针对Docker客户端版本大于1.10.0的用户可以通过修改daemon配置文件/etc/docker/daemon.json来使用加速器
  ``` bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF' { "registry-mirrors": ["https://09sdi3xq.mirror.aliyuncs.com"] }
<EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
  ```
# 安装nvidia-docker
## 添加仓库
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  
sudo systemctl restart docker
  ```
## 拉取镜像使用
  `docker pull ufoym/deepo`

  `docker run --gpus --all image_id`
# 打包镜像
docker多阶段build 实现原理：
  ``` Dockerfile
FROM image1    
RUN something    get intermediate product    
COPY product image1_dir FROM image2    
COPY --from=0 image1_dir image2_dir
  ```
默认情况下，阶段未命名，可以通过整数来引用它们，从第0个FROM指令开始。

可以通过向FROM指令添加as NAME来命名阶段。此示例通过命名阶段并使用COPY指令中的名称来改进前一个示例。停在特定的构建阶段构建镜像时，不一定需要构建整个Dockerfile每个阶段。

可以指定目标构建阶段。以下命令假定使用的是以前的Dockerfile，但在名为builder的阶段停止： `$ docker build --target builder -t alexellis2/href-counter:latest .` 
使用此功能可能的一些非常适合的场景是：
- 调试特定的构建阶段
- 在debug阶段，启用所有调试或工具，而在production阶段尽量精简
- 在testing阶段，应用程序将填充测试数据，但在production阶段则使用生产数据

使用外部镜像作为stage使用多阶段构建时，不仅可以从Dockerfile中创建的镜像中进行复制，还可以使用COPY –from指令从单独的image中复制，使用本地image名称，本地或Docker注册表中可用的标记或标记ID。如有必要，Docker会提取image并从那里开始复制。语法是： `COPY --from=nginx:latest /etc/nginx/nginx.conf /nginx.conf`
# 其他
`docker run -v`

docker挂载目录使用绝对目录，无论是宿主机还是容器，否则挂载后无法同步。