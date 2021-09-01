---
title: 'python多线程'
date: 2019-11-24 17:04:39
tags: [Python]
published: true
hideInList: false
feature: 
isTop: false
---
由于GIL的存在，Python的多线程实际上是伪多线程，在同一时刻只有一个线程。因此Python的多线程适用于IO密集型任务（文件读写，网络请求等），至于计算密集型任务考虑使用多进程。
## 多线程
### 直接调用
  ``` python
import threading

def run(i):
    print(i)
thread = threading.Thread(target=run, args=[1,])
thread.start()
```
### 继承调用
    
  ``` python
class MyThread(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        # run方法必须实现，这里放置自定义的内容
        pass
  ```
### 阻塞和守护线程
thread.join()方法在该线程对象启动了之后调用线程的join()方法之后，那么主线程将会阻塞在当前位置直到子线程执行完成才继续往下走，如果所有子线程对象都调用了join()方法，那么主线程将会在等待所有子线程都执行完之后再往下执行。

setDaemon(True)方法在子线程对象调用start()方法(启动该线程)之前就调用的话，将会将该子线程设置成守护模式启动。当子线程还在运行的时候,父线程已经执行完了，如果这个子线程设置是以守护模式启动的，那么随着主线程执行完成退出时，子线程立马也退出,如果没有设置守护启动子线程(也就是正常情况下)的话，主线程执行完成之后,进程会等待所有子线程执行完成之后才退出。

### 线程锁
  由于多线程共享同一块内存空间，可以访问其中的变量。

  ```python
lock = threading.Lock()
lock.aquire()
# do something
lock.release()
  
# or using with， 锁会自动释放
with lock:
    # do something
  
  ```
死锁发生的情况：  
1）叠加锁：连续调用同一把锁
 ```python
	  lock.aquire()
	  lock.aquire()
	  lock.release()
	  lock.release()
```
python引入RLock来解决这个问题。
2）相互调用锁

### 信号量
  ```python
sem = threading.Semaphore(4)
# 只有4把锁
  
 ```
### 线程池
  ```python
from concurrent.future import ThreadPoolExeutor
  
p = ThreadPoolExecutor(max_workers=10)
  
future = p.submit(fn, args)
data = future.result()
deal_with(data)
# or async deal with data
p.submit(fn, args).add_done_callback(deal_with)
  
future_list = p.map(fn, *iterables)
  
p.shutdown()
  
  ```
对于进程池，只需将ThreadPoolExecuator改为ProcessPoolExecuator即可。
