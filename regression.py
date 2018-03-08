import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # dim?
y = x.pow(2) + 0.2*torch.rand(x.size())

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)     # 神经网络中的一层，隐藏层
        """n_features: 输入个数, n_hidden隐藏层的输出个数"""
        self.predict = torch.nn.Linear(n_hidden, n_output)
        """n_hidden:隐藏层的输出作为预测层的输入; n_output: 输出结果的个数，因为只要求一个y值，所以是1"""

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(1, 10, 1)
# print(net)  # 打印出神经网络层结构

plt.ion()   # 将 matplotlab 设置为实时打印
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)   # optmizer:优化器; SGD:其中一种优化器; lr:学习率
loss_func = torch.nn.MSELoss()  # MESLoss: 均方差

for t in range(10000):
    prediction = net(x)     # 每一步的预测值
    loss = loss_func(prediction, y)     # 计算预测值与真实值的误差，预测值在前，真实值在后

    optimizer.zero_grad()       # 将梯度置为0，因为每次计算损失值都会将梯度保留，所以此处要先清零
    loss.backward()             # 反向传递，计算梯度
    optimizer.step()            # 用optimizer优化梯度

    if t % 5 == 0:      # 每5步打印一次
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)