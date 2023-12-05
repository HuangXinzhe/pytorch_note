# -*- coding:utf-8 -*-
"""
计算图是用来描述运算的有向无环图，有两个主要元素：节点 (Node) 和边 (Edge)。
节点表示数据，如向量、矩阵、张量。边表示运算，如加减乘除卷积等。

叶子节点的概念主要是为了节省内存，在计算图中的一轮反向传播结束之后，非叶子节点的梯度是会被释放的。

非叶子节点的梯度为空，如果在反向传播结束之后仍然需要保留非叶子节点的梯度，可以对节点使用retain_grad()方法。

而 Tensor 中的 grad_fn 属性记录的是创建该张量时所用的方法 (函数)。而在反向传播求导梯度时需要用到该属性。
"""
import torch
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
# y=(x+w)*(w+1)
a = torch.add(w, x)     # retain_grad()
# a.retain_grad()
b = torch.add(w, 1)
y = torch.mul(a, b)
# y 求导
y.backward()
# 打印 w 的梯度，就是 y 对 w 的导数
print(w.grad)

# 查看叶子结点
print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# 查看梯度
print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)

# 查看 grad_fn
print("w.grad_fn = ", w.grad_fn)
print("x.grad_fn = ", x.grad_fn)
print("a.grad_fn = ", a.grad_fn)
print("b.grad_fn = ", b.grad_fn)
print("y.grad_fn = ", y.grad_fn)


"""
pytorch的动态图机制是将运算与搭建同时进行
"""