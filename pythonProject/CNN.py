import torch

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv = torch.nn.Sequential(
            #1.卷积操作
            torch.nn.Conv2d(1,32,kernel_size=5,padding=2),
            #2.归一化BN层
            torch.nn.BatchNorm2d(32),
            #3.非线性激活层 Relu函数
            torch.nn.ReLU(),
            #4.最大池化
            torch.nn.MaxPool2d(2)
        );
        #fc层
        self.fc=torch.nn.Linear(in_features=14*14*32,out_features=10);

    def forward(self,x):
        out = self.conv(x)
        #将图像数据展开成一维
        #x输入的张量（n,c,h,w）
        out = out.view(out.size()[0],-1)
        out = self .fc(out)
        return out