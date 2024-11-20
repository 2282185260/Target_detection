import cv2
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN


test_data = dataset.MNIST(
    root = "mnist",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

test_loader = data_utils.DataLoader(dataset=test_data,
                                     batch_size=64,
                                     shuffle=True)

cnn = torch.load("model/minist_model.pkl")
cnn = cnn.cuda()
loss_test = 0
rightValue = 0
loss_func = torch.nn.CrossEntropyLoss()

for index, (images, labels) in enumerate(test_loader):

    images = images.cuda()
    labels = labels.cuda()
    outputs = cnn(images)
    _,pred =outputs.max(1)
    loss = loss_func(outputs, labels)
    rightValue += (pred == labels).sum().item()

    #batch 64 , c, h, w
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()

    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_data = im_data.transpose(1,2,0)
        im_label = labels[idx]
        im_pred = pred[idx]
        print("预测值为{}".format(im_pred))
        print("真实值为{}".format(im_label))

        cv2.imshow("nowImage",im_data)

        #print(im_data.shape)  # 输出图像维度，包括通道数

        cv2.waitKey(0)

print("loss为{}，准确率是{}".format(loss_test,rightValue / len(test_data)))

