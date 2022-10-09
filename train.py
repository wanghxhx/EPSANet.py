import torch
from torch.utils.tensorboard import SummaryWriter
from conmat import *

tb = SummaryWriter()


def train(dataloader, num_epochs, net, loss_func, optimizer, batch_size):
    total_step = len(dataloader)
    for epoch in range(num_epochs):
        net.train()
        con_mat = 0
        for i, (img, label) in enumerate(dataloader):
            # 将图片和标签放到 GPU 上训练
            img = img.cuda()
            label = label.cuda()

            # 将一个 batch 的图片输入 Unet 网络进行训练
            img_out = net(img)  # 将图片输入网络
            loss = loss_func(img_out, label)  # 计算损失函数

            # # bat_iou = bat_iou + mean_iou
            optimizer.zero_grad()  # 将梯度归零

            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 对网络进行优化
            con_mat = con_mat + confusion_matrix(img_out, label, batch_size)

        # 可视化 loss 的变化
        con_mat = con_mat / total_step  # 一个 epoch 的平均混淆矩阵
        # print(con_mat.type(torch.IntTensor))
        accuracy, f1score, sum_iou = Evaluation(con_mat)
        tb.add_scalar('Loss', loss.item(), epoch)
        tb.add_scalar('IoU', sum_iou, epoch)
        tb.add_scalar('Accuracy', accuracy, epoch)
        tb.add_scalar('F1-score', f1score, epoch)
        print('Epoch [{}/{}], Loss: {:.5f}'
              .format(epoch + 1, num_epochs, loss.item()))

        tb.close()
