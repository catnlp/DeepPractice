# Back Propagation

我理解的反向传播，包含两部分内容：链式法则+动态规划。
链式法则提供了一种计算偏导数的方法，而动态规划则进一步优化了计算，用空间换时间

## 1 链式法则

链式法则是微积分中的求导法则，用来求复合函数的导数。

### 1.1 标量

设 x 是实数，f 和 g 是从实数映射到实数的函数，令 y = g(x), z = f(g(x)) = f(y)

```math
\frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}
```

### 1.2 向量

```math
\frac{\partial z}{\partial x_i} = \sum_{j} \frac{\partial z}{\partial y_j} \frac{\partial y_j}{\partial x_i}
```

## 2 动态规划

动态规划的过程可以看出暴力搜索+空间换时间的优化过程

## 3 简单例子

通过一个简单的例子来说明如何进行反向传播

```math
e = (a + b) * (b + 1)
```

计算图如下：

![image](image/tree-def.png)

令a=2, b=1，计算每个节点值并且利用偏导数的定义，求出不同层之间相邻节点的偏导数关系

![image](image/tree-eval-derivs.png)

利用链式法则我们知道：

```math
\frac{\partial e}{\partial a} = \frac{\partial e}{\partial c} \frac{\partial c}{\partial a}

\frac{\partial e}{\partial b} = \frac{\partial e}{\partial c} \frac{\partial c}{\partial b} + \frac{\partial e}{\partial d} \frac{\partial d}{\partial b} 
```

> (de/da)值等于从(a-c-e)的路径上的偏导值的乘积，而(de/db)等于从b到e的路径(b-c-e)上的偏导值的乘积加上路径(b-d-e)上的偏导值的乘积。

**对于上层节点p和下层节点q，需要找到从q节点到p节点的所有路径，并且对每条路径，求得该路径上的所有偏导数的乘积，然后将所有路径的乘积累加起来得到(dp/dq)**

如果对每个变量单独求导数，有冗余的过程(e-c的路径被访问了两次)。对于神经网络来说更是如此。所以为了减少计算，利用空间换时间，记录每个节点到根节点的导数

**所以反向传播算法对于每一个路径只访问一次就能求顶点对所有下层节点的偏导值**

## 4 逻辑回归

## 5 参考文献

- [如何直观地解释 backpropagation 算法](https://www.zhihu.com/question/27239198/answer/89853077)
- [Calculus on Computational Graphs: Backpropagation](http://colah.github.io/posts/2015-08-Backprop/)