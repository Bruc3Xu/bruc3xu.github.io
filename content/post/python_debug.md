---
title: "Python调试"
date: 2020-09-10T11:11:14+08:00
lastmod: 2020-09-10T11:11:14+08:00
draft: false
keywords: []
description: ""
tags: [python]
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
## 使用PDB调试
代码内调试
  ``` python
import pdb
  
# pause here
pdb.set_trace()

```
或者命令行启动`python -m pdb my_script.py`
    
- c : 继续执行 
- w : 显示当前正在执行的代码行的上下文信息 
- a : 打印当前函数的参数列表 
- s : 执行当前代码行，并停在第一个能停的地方 
- n : 继续执行到当前函数的下一行，或者当前行直接返回 
- q: 退出调试，程序执行会被终止

## gdb调试
gdb调试主要为了一些Python无法捕捉的错误（coredump错误），这些错误可能由于调用C/C++代码引起。
1. `ulimit -c unlimited` 在当前目录下生成core文件（或者选择其他目录）。
2. `gdb python core` 进行调试，调试命令与上文相同。
## python cprofile性能测试
```python
import cProfile


# Code containing multiple dunctions
def create_array():
  arr=[]
  for i in range(0,400000):
    arr.append(i)

def print_statement():
  print('Array created successfully')


def main():
  create_array()
  print_statement()


if __name__ == '__main__':
    cProfile.run('main()')

```
输出结果：
- ncalls : 调用次数
- tottime: 函数执行总时间，不包含子函数
- percall: 每次调用时间
- cumtime: 累计时间，包含从函数开始到结束的时间（包括子函数或者递归）

使用Profile类：
```python
def create_array():
  arr=[]
  for i in range(0,400000):
    arr.append(i)

def print_statement():
  print('Array created successfully')


def main():
  create_array()
  print_statement()

if __name__ == '__main__':
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.print_stats()
```

使用装饰器：

``` python
import cProfile
import os
import pstats
import random


def do_cprofile(filename):
    """Decorator for function profiling."""
    def wrapper(func):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            # Sort stat by internal time.
            sortby = "tottime"
            # 导出数据
            ps = pstats.Stats(profile).sort_stats(sortby)
            ps.dump_stats(filename)
            return result

        return profiled_func

    return wrapper


@do_cprofile("result.prof")
def search_function():
    data = [random.randint(0, 99) for p in range(0, 1000)]
    for i in data:
        if i > 80:
            pass


if __name__ == '__main__':
    search_function()
```

### 可视化
安装snakeviz，`pip install snakeviz`。

执行`snakeviz result.prof`。

在web页面可以看到结果如图。

![](/post/snakeviz_pf.png)