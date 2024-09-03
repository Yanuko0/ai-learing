import  torch
# from requests.packages import target
# from torch.xpu import device
# print(torch.__version__)
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import  torch.optim as optim

#1. 檢測cuda是否可用
use_cuda = torch.cuda.is_available()
# print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")

#2. 如果可用設置device變量
# if use_cuda:
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")

#4. 設置對數據進行處理的邏輯,Compose是組合
transform = transforms.Compose([
    # 讓數據轉換成Tensor張量
    transforms.ToTensor(),
    # 讓圖片進行標準歸一化,0.1307 是標準歸一化的均值, 0.3081對應標準歸一化的方差
    transforms.Normalize((0.1307,),(0.3081,))
])

#3. 讀取數據
# 創建一個data文件夾,裡面放下載的內容
datasets1 = datasets.MNIST("./data", train=True, download=True , transform=transform)
datasets2 = datasets.MNIST("./data", train=False, download=True , transform=transform)

# 5.設置數據的加載器, 順帶手設置批次大小和是否打亂數據順序
train_loader = torch.utils.data.DataLoader(datasets1, batch_size=128, shuffle = True)
test_loader = torch.utils.data.DataLoader(datasets2, batch_size = 1000)

# 6.
# for batch_idx, data in enumerate(train_loader, 0):
#     inputs, target = data
#     # view在下一行會把我們的訓練集(60000,1,28*28)轉換成(60000*1,28*28)
#     x = inputs.view(-1, 28*28)
#     # 計算所有訓練樣本的標準差和均值
#     x_std = x.std().item()
#     x_mean = x.mean().item()
#
# print("均值mean為:" + str(x_mean))
# print('標準差std為' + str(x_std))

#7. 通過自定義類來構件模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

#8. 創建一個模型實例
model = Net().to(device)

# 定義訓練模型的邏輯
def train_step(data, target, model, optimizer):
    optimizer.zero_grad()
    output = model(data)
    # nll代表著negative log likely hood 負對數似然
    loss = F.nll_loss(output, target)
    # 反向傳播的本質是去求梯度
    loss.backward()
    # 本質是應用梯度去調參
    optimizer.step()
    return loss

# 定義使用模型的邏輯
def test_step(data, target, model, test_loss, correct):
    output = model(data)
    # 累積的批次損失
    test_loss += F.nll_loss(output, target, reduction='sum').item()
    # 獲得對數概率最大值對應的索引號, 其實就是類別號
    pred = output.argmax( dim=1, keepdim=True )
    correct += pred.eq(target.view_as(pred)).sum().item()
    return test_loss, correct

# 創建訓練調參使用的優化器
optimizer = optim.Adam(model.parameters(), lr=0.001 )

# 真正的分輪次訓練
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss = train_step(data, target, model, optimizer)
        # 每隔10個批次打印信息
        if batch_idx % 10 == 0:
            print('Train Epoch:{} [{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))

        model.eval()
        test_loss = 0
        correct = 0
        for epoch in range(EPOCHS):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                loss = train_step(data, target, model, optimizer)
                if batch_idx % 10 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))