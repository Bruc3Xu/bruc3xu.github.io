---
title: "Pybind11 Cmake tutorial"
date: 2020-08-25T14:32:08+08:00
lastmod: 2020-09-08T14:32:08+08:00
draft: false
keywords: []
description: ""
tags: [python, c++]
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
# 使用pybind11和CMake来构建C++/Python项目
期望最终结果：
* 独立于`pybind11`的`C++`项目。
* 基于`C++` wrapper的`Python`库。
* 都使用`CMake`来build项目。


项目的整体结构如下
```txt
├── cmake_example
│   └── __init__.py
├── CMakeLists.txt
├── cpp
│   ├── a.cpp
│   ├── CMakeLists.txt
│   ├── include
│   │   ├── a.h
│   │   ├── b.h
│   │   └── define.h
│   ├── libs
│   │   └── libb_lib.so
│   └── third_party
│       ├── b.cpp
│       ├── build.sh
│       └── CMakeLists.txt
├── README.md
├── setup.py
├── src
│   └── main.cpp
└── tests
    └── test.py
```
源码参见[https://github.com/FireLandDS/pybind11_cmake_demo](https://github.com/FireLandDS/pybind11_cmake_demo)。


这里我们需要wrap两个库a和b，假设库a源码已知，库b源码未知（通常第三方库）。

首先我们需要生成a库，CMakeLists如下：
```CMake
cmake_minimum_required(VERSION 3.4...3.18)
project(a LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(include)
add_library(a_lib SHARED a.cpp)
```
a的源码如下
```c++
//define.h
#ifndef UTILS_H_
#define UTILS_H_
#define PI 3.1415926
#if defined(WIN32) || defined(_WIN64)

#ifdef LIB_EXPORTS
#define LIB_API extern "C" __declspec(dllexport)
#else
#define LIB_API extern "C" __declspec(dllimport)
#endif // LIB_EXPORTS

#elif defined __linux__

#ifndef LIB_API
#define LIB_API extern "C"
#endif // LIB_API

#endif // defined (WIN32) || defined (_WIN64)

#endif


//a.h
#include "define.h"
LIB_API double add_PI(double a, double b);

//a.cpp
#include "a.h"
LIB_API double add_PI(double a, double b) { return a + b + PI; }

```

b假设不知道源码，这里我们先生成动态库文件，然后放在cpp/libs中。
```c++
//b.h
#include "define.h"

LIB_API double divide(double a, double b);

//b.cpp
#include "../include/b.h"


LIB_API double divide(double a, double b)
{
  if (b == 0.0)
    return -1.0;
  return a / b;
}

```

执行
```bash
cd cpp/third_party
cmake -S . -B build
cmake --build build
```
在build文件夹中得到libb_lib.so文件（或者dll）放置到libs中。

然后我们来看如何wrap c++代码。
```c++
#include <pybind11/pybind11.h>
#include "a.h"
#include "b.h"


int add(int i, int j)
{
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(cmake_example, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("add_pi", &add_PI);
    m.def("divide", &divide);

    m.def(
        "subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

}
```

PYBIND11_MODULE是最重要的宏，定义了我们生成的python库。
其中，cmake_example是库名，不需要引号。
m可以理解成模块对象，任意命名，需要与内部的相对应，m.doc表示help说明，m.def用来注册函数，支持匿名函数。

注意事项：c++中存在string引用，int引用等，而这在python中是无法实现的，因为这些类型在python中是常量值。

最后，我们看一下主CMakeLists
```cmake
cmake_minimum_required(VERSION 3.4...3.18)
project(cmake_example)

include_directories(${CMAKE_SOURCE_DIR}/cpp/include)

add_subdirectory(pybind11)
add_subdirectory(cpp)

pybind11_add_module(cmake_example src/main.cpp)

find_library(b NAMES libb_lib.so HINTS ${CMAKE_SOURCE_DIR}/cpp/libs)
target_link_libraries(cmake_example PRIVATE ${b})
target_link_libraries(cmake_example PRIVATE a_lib)
```

在setup.py中，我们根据CMakeLists做到了自动打包，安装使用
```bash
python setup.py install
或者
pip install .
```
注意事项：
- setup.py中cmake参数有`DCMAKE_INSTALL_RPATH=$ORIGIN/`这一项，可以保证生成的库可以再当前目录找到依赖库。
- 有些第三方库会依赖其他库，在其本身无rpath属性的情况下，可以使用`patchelf --set-rpath`来手动修改。


测试是否正确可以在安装完成后尝试运行`python tests/test.py`.
