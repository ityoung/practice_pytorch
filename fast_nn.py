import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 假数据
n_data = torch.ones(100, 2)         # 数据的基本形态
x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, 1)

# 注意 x, y 数据的数据形式是一定要像下面一样, 即样本为float，标签为int (torch.cat 是在合并数据)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

# method 1
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

net1 = Net(2, 10, 2)     # 输入2个特征，输出2个特征
print(net1)  # 打印出神经网络层结构

# method 2, 即本章的快速创建神经网络的方法
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)
print(net2)

plt.ion()   # 将 matplotlab 设置为实时打印
plt.show()

optimizer = torch.optim.SGD(net2.parameters(), lr=0.002)   # optmizer:优化器; SGD:其中一种优化器; lr:学习率
loss_func = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss:

for t in range(10000):
    out = net2(x)     # 每一步的预测值
    loss = loss_func(out, y)     # 计算预测值与真实值的误差，预测值在前，真实值在后

    optimizer.zero_grad()       # 将梯度置为0，因为每次计算损失值都会将梯度保留，所以此处要先清零
    loss.backward()             # 反向传递，计算梯度
    optimizer.step()            # 用optimizer优化梯度

    if t % 5 == 0:      # 每5步打印一次
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
# plt.show()