from __future__ import print_function
from torch.autograd import Variable
import torch


tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True) # 会计算梯度

t_out = torch.mean(tensor*tensor)               # x^2
v_out = torch.mean(variable*variable)           # mean: 均值, (1+4+9+16)/4 = 7.5

print(t_out)
print(v_out)

vv_out = v_out*5
print(vv_out)
# print(variable)
vv_out.backward()
# print(variable)

print(variable.grad)                            # grad: 梯度
# print(type(variable))
print(variable.data)                            # data: 存储的Tensor数据
print(variable.grad_fn)                            # data: 存储的Tensor数据

# variable.data = tensor
# x = torch.randn(3)
# print(x)
# x = Variable(x, requires_grad=True)
# y = x * 2
# while y.data.norm() < 1000:
#     y = y * 2
# print(y)
#
# gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
# y.backward(gradients)
#
# print(x.grad)
# # x = torch.FloatTensor([0.1, 1.0, 0.0001])
# # print(x)