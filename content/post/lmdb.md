---
title: "LMDB数据库使用"
date: 2020-03-11T14:29:51+08:00
lastmod: 2020-03-11T14:29:51+08:00
draft: false
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
LMDB的全称是Lightning Memory-Mapped Database(快如闪电的内存映射数据库)，它的文件结构简单，包含一个数据文件和一个锁文件，data.mdb, lock.mdb
## LMDB基本操作
### 1.生成一个空的数据库文件

  ``` python
import lmdb

env = lmdb.open("./data", map_size=1099511627776)
# 如果./data不存在则会创建新建目录，并会在下面生成data.mdb,lock.mdb。如果存在，不会覆盖。
# map_size对应最大存储容量，以kb为单位，定义1TB。
env.close()

  ```
### 2.lmdb文件的添加、修改、删除

  ``` python
env = lmdb.open("./train", map_size=1099511627776)

# 参数write设置为True才可以写入
txn = env.begin(write=True)

# 添加数据和键值 
txn.put(key = '1', value = 'aaa') 
txn.put(key = '2', value = 'bbb') 
txn.put(key = '3', value = 'ccc') 

# get函数通过键值查询数据 
print(txn.get(str(2)))

# 通过cursor()遍历所有数据和键值 
for key, value in txn.cursor(): 
    print(key, value) 

# 通过键值删除数据 
txn.delete(key = '1') 

# 修改数据 
txn.put(key = '3', value = 'ddd') 

# 通过commit()函数提交更改 
txn.commit() 
env.close()

  ```
## 图片数据的读写
  ``` python
import numpy as np
import lmdb
import cv2

n_samples= 2


def create_random_image(filename):
    img = (np.random.rand(100,120,3)*255).astype(np.uint8)
    cv2.imwrite(filename, img)

def write_lmdb(filename):
    print('Write lmdb')

    lmdb_env = lmdb.open(filename, map_size=int(1e9))

    X = cv2.imread('random_img.jpg')
    y = np.random.rand(1).astype(np.float32) * 10.0

    for i in range(n_samples):
        with lmdb_env.begin(write=True) as lmdb_txn:
            lmdb_txn.put(('X_'+str(i)).encode(), X)
            lmdb_txn.put(('y_'+str(i)).encode(), y)

            print('X.shape:',X.shape)
            print('y:',y)


def read_lmdb(filename):
    print('Read lmdb')

    lmdb_env = lmdb.open(filename)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    #also can do it without iteration via txn.get('key1')?

    n_counter=0
    with lmdb_env.begin() as lmdb_txn:
        with lmdb_txn.cursor() as lmdb_cursor:
            for key, value in lmdb_cursor:  
                print(key.decode())
                if('X' in key):
                    # 注意dtype要与之前的相同，否则解码错误
                    print('X.shape', np.frombuffer(value, dtype=np.uint8).shape)
                if('y' in key):
                    print(np.fromstring(value, dtype=np.float32))

                n_counter=n_counter+1

    print('n_samples',n_counter)


def write_lmdb_jpg(filename):
    print 'Write lmdb'

    lmdb_env = lmdb.open(filename, map_size=int(1e9))

    X= cv2.imread('random_img.jpg')
    y= np.random.rand(1).astype(np.float32)*10.0

    for i in range(n_samples):
        with lmdb_env.begin(write=True) as lmdb_txn:
            lmdb_txn.put('X_'+str(i), cv2.imencode('.jpg', X)[1])
            lmdb_txn.put('y_'+str(i), y)

            print 'X.shape', cv2.imencode('.jpg', X)[1].shape
            print 'y:',y


def read_lmdb_jpg(filename):
    print('Read lmdb')

    lmdb_env = lmdb.open(filename)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()

    #also can do it without iteration via txn.get('key1')?

    n_counter=0
    with lmdb_env.begin() as lmdb_txn:
        with lmdb_txn.cursor() as lmdb_cursor:
            for key, value in lmdb_cursor:
                if('X' in key):
                    X_str= np.fromstring(value, dtype=np.uint8)
                    print('X_str.shape', X_str.shape
                    X= cv2.imdecode(X_str, cv2.CV_LOAD_IMAGE_COLOR))
                    print('X.shape', X.shape)
                if('y' in key):
                    print(np.fromstring(value, dtype=np.float32))

                n_counter=n_counter+1

    print('n_samples',n_counter)

create_random_image('random_img.jpg')

#Write as numpy array       
write_lmdb('temp.db')
read_lmdb('temp.db')

#Write as encoded jpg
write_lmdb_jpg('temp_jpg.db')
read_lmdb_jpg('temp_jpg.db')

```