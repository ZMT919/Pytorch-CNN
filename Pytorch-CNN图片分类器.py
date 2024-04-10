# 使用Pytorch构建一个分类器
# 将不同的图像进行分类的神经网络分类器，对输入的图片进行判别并完成分类
# 本案例采用CIFAR10数据集
# 每张图片的尺寸是3*32*32,代表彩色3通道，32*32尺寸大小的图像
# CIFAR10数据集总共有10种不同的种类，分别为
# airplane,automobile,bird,cat,deer,
# dog,frog,horse,ship,truck

# 训练分类器的步骤
# 1.使用torchvision下载CIFAR10数据集
# 2.定义卷积神经网络
# 3.定义损失函数
# 4.在训练集上训练模型
# 5.在测试集上测试模型

# 分类器构建此处开始...
# --------------------------------------------------
# 1.使用torchvision下载CIFAR10数据集
# 导入torchvision包辅助下载数据集
import torch
import torchvision
import torchvision.transforms as transforms

# 下载数据集并对图片进行调整，因为torchvision数据集输出的是PILImage格式，
# 数据域在[0, 1],我们将其转换为标准数据域[-1, 1]的张量格式
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)
                                       # 此处num_workers在linux系统中可以改为2，表示多线程
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=True, num_workers=0)
                                       # 此处num_workers在linux系统中可以改为2，表示多线程
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 展示若干训练集图片
import matplotlib.pyplot as plt
import numpy as np

# 构建展示图片的函数
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 从数据迭代器中读取一张图片
dataiter = iter(trainloader)
images, labels = dataiter.__next__()

# 展示图片
imshow(torchvision.utils.make_grid(images))
# 打印标签label
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# 2.定义卷积神经网络
# 采用3通道
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义两个卷积层
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 定义三个全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 变换x的形状以适配全连接层的接入
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
print(net)

# 3.定义损失函数
import torch.optim as optim
# 选择交叉熵的损失函数
criterion = nn.CrossEntropyLoss()
# 选择随机梯度下降优化器
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 4.在训练集上训练数据
# 采用基于梯度下降的优化算法，都需要很多个轮次的迭代训练
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # data中包含输入图像张量inputs, 标签张量labels
        inputs, labels = data
        # 首先将优化器梯度归为0
        optimizer.zero_grad()
        # 输入图像张量进行网络，得到输出张量outputs
        outputs = net(inputs)
        # 利用网络的输出outputs和标签labels计算损失值
        loss = criterion(outputs, labels)
        # 反向传播+参数更新，是标准代码的标准流程
        loss.backward()
        optimizer.step()
        # 打印轮次和损失值
        running_loss += loss.item()
        if(i + 1) % 2000 == 0:
            print('[%d, %5d] loss:%.3f'%
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

# 保存模型
# 首先设定模型的保存路径
PATH = './cifar_net.pth'
# 保存模型的状态字典
torch.save(net.state_dict(), PATH)

# 5.在测试集上测试模型
# 5.1展示测试集中的若干图片
dataiter = iter(testloader)
images, labels = dataiter.__next__()
# 打印原始图片
imshow(torchvision.utils.make_grid(images))
# 打印真实标的标签
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 加载模型并对测试图片进行预测
# 首先实例化模型的类对象
net = Net()
net.load_state_dict(torch.load(PATH))
# 利用模型对图片进行预测
outputs = net(images)
# 共有10个类别，采用模型计算出的概率最大的作为预测的类别
_, predicted = torch.max(outputs, 1)
# 打印预测标签的结果
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# 测试全部测试集的表现
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' %(100 * correct / total))

# 按类别进行计算准确率,即在哪些类别上表现更好，在哪些类别上表现差。计算在各个类别上的准确率
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' %(classes[i], 100 * class_correct[i] / class_total[i]))

