---
title: "Python Dataclass"
date: 2020-10-10T13:59:38+08:00
lastmod: 2020-10-10T13:59:38+08:00
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
dataclass是python3.7的新特性，是一个主要包含数据的类。
<!--more-->

```python
from dataclasses import dataclass, make_dataclass

@dataclass
class DataClassCard:
  rank: str
  suit: str


DataClassCard2 = make_dataclass('DataClassCard2', ['rank', 'suit'])

>>> queen_of_hearts = DataClassCard('Q', 'Hearts')
>>> queen_of_hearts.rank
'Q'
>>> queen_of_hearts
DataClassCard(rank='Q', suit='Hearts')
>>> queen_of_hearts == DataClassCard('Q', 'Hearts')
True

```

如果我们使用一般方法来创建一个类，
```python
class RegularCard:
  def __init__(self, rank, suit):
    self.rank = rank
    self.suit = suit

>>> queen_of_hearts = RegularCard('Q', 'Hearts')
>>> queen_of_hearts.rank
'Q'
>>> queen_of_hearts
<__main__.RegularCard object at 0x7fb6eee35d30>
>>> queen_of_hearts == RegularCard('Q', 'Hearts')
False

```

python使用dataclass创建的类，内部已经重写了`__init__, __repr__, __eq__`等方法。


如果我们要实现这些功能

```python
class RegularCard
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(rank={self.rank!r}, suit={self.suit!r})')

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented
        return (self.rank, self.suit) == (other.rank, other.suit)
```

dataclass可以看做是对namedtuple的替代
```python
from collections import namedtuple

NamedTupleCard = namedtuple('NamedTupleCard', ['rank', 'suit'])

>>> queen_of_hearts = NamedTupleCard('Q', 'Hearts')
>>> queen_of_hearts.rank
'Q'
>>> queen_of_hearts
NamedTupleCard(rank='Q', suit='Hearts')
>>> queen_of_hearts == NamedTupleCard('Q', 'Hearts')
True
```

namedtuple有很多限制，因为其本身实质上是tuple，无法修改，无法设置默认值，无法继承等。


设置默认值和类型提示
```python
from dataclasses import dataclass

@dataclass
class CName:
    name: str


@dataclass
class Position:
    name: str
    lon: float = 0.0
    lat: float = 0.0
    names: List[CName]
```

默认工厂函数（dataclass可变数据类型必须使用）

```python

@dataclass
class D:
    x: list = field(default_factory=list)

assert D().x is not D().x
```