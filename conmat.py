import torch


def confusion_matrix(lab_pre, lab_tru, batch_size):
    d = 0
    iou_value = 0
    for j in range(batch_size):
        # 去掉图像的 batch 维度
        label_pred = lab_pre[j].squeeze()
        label_true = lab_tru[j].squeeze()

        # 将图像转化成二值化图像
        label_pred[(label_pred < 0.5)] = 0
        label_pred[(label_pred >= 0.5)] = 1

        label_true[(label_true == 255)] = 1

        num_class = 2  # 道路和背景属于二分类问题

        # mask 拉成 1-d tensor，其中包含所有符合条件的值
        mask = (label_pred >= 0) & (label_pred < num_class)
        a = label_true[mask].type(torch.IntTensor)
        b = label_pred[mask].type(torch.IntTensor)

        c = a * num_class + b  # 真实类别作为行，乘以类别数加预测值就是其在混淆矩阵的位置减一
        d = d + torch.bincount(c, minlength=(num_class * num_class)).reshape(num_class, num_class)

    # print(d/batch_size)
    # print(iou_value/batch_size)
    return d / batch_size


# 求相应混淆矩阵的评价指数
# 应用于二分类
def Evaluation(con_mat):
    accuracy = (con_mat[1][1] + con_mat[0][0])/(con_mat[1][0] + con_mat[0][1] + con_mat[1][1] + con_mat[0][0])  # 准确度
    precision = con_mat[1][1] / (con_mat[1][0] + con_mat[1][1])  # 精度
    recall = con_mat[1][1] / (con_mat[0][1] + con_mat[1][1])  # 用一个 epoch 的平均混淆矩阵求平均 Recall（召回率）
    f1score = (2 * precision * recall)/(precision + recall)  # f1 score
    iou = con_mat[1][1] / (con_mat[1][0] + con_mat[0][1] + con_mat[1][1])  # 用一个 epoch 的平均混淆矩阵求平均 IoU

    return accuracy, f1score, iou


