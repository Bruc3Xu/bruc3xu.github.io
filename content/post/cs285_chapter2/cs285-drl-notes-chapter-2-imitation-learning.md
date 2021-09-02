---
title: 'cs285 DRL notes chapter 2: imitation learning'
date: 2020-08-23 15:13:48
tags: [reinforcement learning,cs285]
published: true
hideInList: false
isTop: false
---
模仿学习是一种监督学习方法，行为克隆是其中的一类方法。其基本思想是从专家演示数据中学习到一个尽可能接近专家策略的行为策略。我们的数据集是依据专家策略采样得到的$o_t, a_t$，可以认为是输入和标签。

强化学习与监督学习的区别在于：
1. 强化学习中，数据并不是独立同分布的（i.i.d）。
2. 强化学习并没有准确的标签，只有奖励值这一弱监督信号。

模仿学习的一个问题是泛化能力差。例如，在状态$s_t$，智能体做出了一个错误决策（因为学习得到的策略分布与专家策略分布不能完全相同，这个问题是无法避免的），到达一个新的状态$s_t'$。这个状态对于智能体来说是没有见过的，即没有学习到的，那么智能体就会选择一个随机的动作，偏离学习到的轨迹。整个过程如下图所示。
![](/post/cs285_chapter2/distribution_shift.png)

## DAgger(Dataset Aggregation)
DAgger的思想是：既然没有见过的状态不在原有分布之内，那么我们使用专家策略对这个状态进行动作选择，并将其加入数据集进行训练，那么就可以解决上文提到的分布偏移问题。
***
Human data D = {o_1,a_1,...,o_N,a_N}\
While not Converged\
    $\quad$ Train $\pi_\theta(a_t|o_t)$ from human data $\mathcal{D} = {o_1,a_1,...,o_N,a_N}$\
    $\quad$ Run $\pi_\theta(a_t|o_t)$ to get dataset $\mathcal{D}_\pi = {o_1,...,o_M}$\
    $\quad$ Ask human to label $\mathcal{D}_\pi$ with actions $a_t$\
    $\quad$ Aggregate $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D_\pi}$\
Return optimal imitation-learned trajectory as $\tau^{return}$
***
DAgger算法存在的问题：
1. 无法保证人类标记数据的可靠性
2. 人类做出决策不仅依赖当前状态，还依赖历史状态（并不一定指上一个状态），不满足马尔可夫性质。

## 模仿学习存在的为题和解决方法
### 行为不具有马尔可夫性质
这个问题即DAgger算法的第二个问题。在这种情况下，策略学习由$\pi_{\theta}(a_t|o_t)$变为
$\pi_{\theta}(a_t|o_t, o_{t-1}, ...)$。

一种解决方法是将历史状态堆叠起来作为新的状态。
```python
class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.stack(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        # typically image observation
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k))

    def _reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))
```

另一种方法是使用循环神经网络来处理时序输入，一般使用LSTM网络。

但更多的历史状态作为输入很容易导致**因果混淆**。例如，当一辆自动驾驶车辆遇到障碍物，踩下刹车时，刹车灯会亮起。智能体更容易通过刹车灯的亮灭而不是障碍物来进行决策。DAgger因为使用人为标记来扩充数据集，不会出现此类问题。

### 多模态行为
专家策略可能会出现多模态行为，例如，针对车辆前方的障碍物，可能会选择向左变道或者向右变道。这时，如果学习的策略是一个概率分布，那么在这个状态下的动作就会被平均，趋向于不变道，这显然是错误的。
解决方法：
1. 使用多个分布的累加代替单一策略分布，一般使用混合高斯分布。
2. 将状态映射到latent空间。
3. 使用自回归离散化 (Autoregressive Discretization)。
## 误差分析
假设我们可以寻找到一个策略$\pi$ ，其与最优策略的损失函数值小于给定的精度$\epsilon$，那么我们可以证明出来，这个策略与专家策略的决策质量上有如下的保证：
$$
V(\pi_E)-V(\pi)<=\frac{2\sqrt{2}}{(1-\gamma)^2}\sqrt{\epsilon}
$$
可以看到，损失函数值越小，两者的值函数差异越小。但与此同时，我们注意到这个差异是以$1/(1-\gamma)^2$的速度在放大。这个现象在模仿学习中被称作为”复合误差“ (compounding errors）：对于一个有效决策长度（以$1-\gamma$来衡量， $\gamma$越接近1，有效决策长度越长）的模仿学习任务，值函数值差异随目标函数值差异以二次方的速度增长。也就说：对于有效决策长度比较长的任务来讲，即使我们把目标函数优化地很小，值函数的差异依然可能很大。这个结论在以前的paper[<sup>1, 2</sup>](#refer-anchor)和最近的paper[<sup>3</sup>](#refer-anchor)里都有详细的描述。

由于数据增广和环境交互，DAgger 算法会大大减小未访问的状态的个数，从而减小复合误差。

## 总结
总结来说，模仿学习通常有一定局限性（分布不匹配的问题、误差累积问题），但有时候能做得不错，如使用一些稳定控制器，或者从稳定轨迹分布中抽样，亦或是使用DAgger之类的算法增加更多的在线数据，理想化地如使用更好的模型来拟合得更完美。

更进一步的关于模仿学习的内容，可以参考[Imitation Learning](https://www.lamda.nju.edu.cn/xut/Imitation_Learning.pdf)这本书。
## 参考
<div id="refer-anchor"></div>
[1]Ross, Stéphane, Geoffrey Gordon, and Drew Bagnell. "A reduction of imitation learning and structured prediction to no-regret online learning." Proceedings of the 14th international conference on artificial intelligence and statistics. 2011.

[2]Syed, Umar, and Robert E. Schapire. "A reduction from apprenticeship learning to classification."Advances in neural information processing systems. 2010.

[3]Xu, Tian, Ziniu Li, and Yang Yu. "Error Bounds of Imitating Policies and Environments." Advances in neural information processing systems. 2020.
