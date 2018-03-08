import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5, 5, 200)
x = Variable(x)
# print(x)
x_np = x.data.numpy()
# print(x_np)
y_relu = F.relu(x).data.numpy()
# print(y_relu)

plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1,5))
plt.legend(loc='best')

plt.show()