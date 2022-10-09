from torch.utils.data import Dataset
import torchvision.transforms as transform
from EPSANet import *
from dataset import *
from train import *
from test import *
from loss import *


if __name__ == '__main__':
    net = EPSANet(EPSABlock, [3, 4, 6, 3]).cuda()

    # 超参设置
    batch_size = 5
    num_epochs = 10

    learning_rate = 0.1

    # 数据增强方法
    transform = torchvision.transforms.Compose([
        transform.Resize(256),
        transform.ToTensor()
    ])

    # 数据输入输出路径
    input_path = "C:/Users/wangh/mmsegmentation/tools/data/Mass/test/image"
    label_path = "C:/Users/wangh/mmsegmentation/tools/data/Mass/test/label"
    test_image_path = "C:/Users/wangh/mmsegmentation/tools/data/Mass/test/image"
    test_label_path = "C:/Users/wangh/mmsegmentation/tools/data/Mass/test/label"

    # Dataset Dataloader 设置
    train_data = MyDataset(transform, input_path, label_path)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0,
                                             drop_last=True)
    test_image = TestDataset(transform, test_image_path, test_label_path)
    test_dataloader = torch.utils.data.DataLoader(test_image, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    # 损失函数和优化器设置
    optimizer = torch.optim.Adam(net.parameters())
    loss_func = nn.BCEWithLogitsLoss()

    # 训练
    train(dataloader, num_epochs, net, loss_func, optimizer, batch_size)
    # 测试
    test(test_dataloader, net)
