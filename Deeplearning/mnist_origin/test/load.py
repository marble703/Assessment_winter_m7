import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



batch_size_test = 50   #测试时的批次大小，调整没啥影响，一般和训练批次大小相似

log_interval = 100     #日志输出间隔，控制输出训练信息频率

random_seed = 42 #随机种子，用于复现结果
torch.manual_seed(random_seed) #设置随机种子

#使用确定性算法确保结果可复现
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 检查是否有可用的CUDA,有则使用
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/',
                               train=False,   #加载测试集
                               download=True,
                               transform=torchvision.transforms.ToTensor(),),
    batch_size = batch_size_test,
    shuffle=True)

#定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() #初始化方法
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5) #卷积层1：输入通道数，输出通道数，kernel_size表示卷积核的大小
        self.conv2 = nn.Conv2d(8, 64, kernel_size=5) #卷积层2
        self.fc1 = nn.Linear(1024, 256) #全连接层类1:输入特征数，输出特征数
        self.fc2 = nn.Linear(256, 10) #全连接层类2
        self.conv_drop = nn.Dropout2d() #二维Dropout层，正则化，防止过拟合
        
    def forward(self, x):     #前向传播方法
        x = F.relu(F.max_pool2d(self.conv1(x), 2))         
        x = F.relu(F.max_pool2d(self.conv_drop(self.conv2(x)), 2))
        x = x.view(-1, 1024)        #将二维张量x展平为一维张量。使用-1可以根据其他维度推断出一个维度的大小，1024是展平后张量的大小。
        x = F.relu(self.fc1(x))     #对展平后的张量应用全连接层1，对全连接层1的结果应用ReLU激活函数 
        x = self.fc2(x)             #对张量x应用全连接层2
        return F.log_softmax(x, dim = 1)

load_path = 'model.pt'
network = Net()
network = torch.load(load_path)

def test(epoch):
    network.eval() #将神经网络设置为评估（测试）模式
    test_loss = 0  #初始化测试损失，用于计算测试集上的累积损失
    correct = 0    #初始化正确预测的样本数，用于计算准确率
    with torch.no_grad(): #上下文管理器，在该块中的操作不会影响到梯度计算
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    return (correct / len(test_loader.dataset))

print("Accuracy: " + str(test(1).item()))
