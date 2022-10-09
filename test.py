import numpy as np
import matplotlib as plt
from PIL import Image
from conmat import *


# 测试函数
def test(test_dataloader, net):
    con_mat = 0
    total_step = len(test_dataloader)
    for i, (image, label) in enumerate(test_dataloader):
        img = image.cuda()
        img_out = net(img)
        img_out = img_out.cpu()
        con_mat = con_mat + confusion_matrix(img_out, label, 1)

        array_img = img_out.detach().numpy()  # transfer tensor to array
        array_show = np.squeeze(array_img[0], 0)  # extract the image being showed

        array_show[array_show >= 0.5] = 255
        array_show[array_show < 0.5] = 0
        im = Image.fromarray(array_show)
        image_name = "./result/test_" + str(i+1) + ".jpg"
        if im.mode == "F":
            im = im.convert('RGB')
        im.save(image_name)

    con_mat = (con_mat/total_step).type(torch.IntTensor)
    print(con_mat)