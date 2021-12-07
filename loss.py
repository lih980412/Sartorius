import torch
from typing import List, Tuple
import torch.nn.functional as F
from torch.autograd import Variable


# 19 1
def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    # cond = n < beta
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


# 25 1
def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits : 预测类别概率信息，shape=[num_anchors, num_classes]   512*80
        box_regression : 预测边目标界框回归信息
        labels : 真实类别信息
        regression_targets : 真实目标边界框信息

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # 计算类别损失信息

    # classification_loss = F.cross_entropy(class_logits, labels)
    classification_loss = MultiCEFocalLoss(class_logits, labels, class_num=81, alpha=0.25, gamma=2)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    # 返回标签类别大于0的索引
    # sampled_pos_inds_subset = torch.nonzero(torch.gt(labels, 0)).squeeze(1)
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]

    # 返回标签类别大于0位置的类别信息
    labels_pos = labels[sampled_pos_inds_subset]

    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    # 计算边界框损失信息
    box_loss = smooth_l1_loss(
        # 获取指定索引proposal的指定类别box信息
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    ) / labels.numel()

    return classification_loss, box_loss


def MultiCEFocalLoss(predict, target, class_num, alpha=None, gamma=2, reduction=None):
    if alpha is None:
        alpha = Variable(torch.ones(class_num, 1))
    else:
        alpha = alpha
    eps = 1e-7
    reduction = "mean"
    class_mask = F.one_hot(target, class_num)
    # y_pred = predict.view(predict.size()[0], predict.size()[1])
    y_pred = F.softmax(predict, dim=1)

    target = class_mask.view(y_pred.size())
    ce = -1 * torch.log(y_pred + eps) * target
    floss = torch.pow((1 - y_pred), gamma) * ce
    floss = torch.mul(floss, alpha)
    if reduction == 'mean':
        loss = floss.sum(1).mean()
    elif reduction == 'sum':
        loss = floss.sum(1).sum()
    return loss


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)  # sigmoid获取概率
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (
                1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

# class MultiCEFocalLoss(torch.nn.Module):
#     def __init__(self, class_num, gamma=2, alpha=None):
#         super(MultiCEFocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             self.alpha = alpha
#         self.gamma = gamma
#         self.class_num = class_num
#         self.eps = 1e-7
#
#     def forward(self, predict, target):
#
#         class_mask = F.one_hot(target, self.class_num)
#         # y_pred = predict.view(predict.size()[0], predict.size()[1])
#         y_pred = F.softmax(predict, dim=1)
#
#         target = class_mask.view(y_pred.size())
#         ce = -1 * torch.log(y_pred + self.eps) * target
#         floss = torch.pow((1 - y_pred), self.gamma) * ce
#         floss = torch.mul(floss, self.alpha)
#         floss = torch.sum(floss, dim=1)
#         # print(floss)
#         # print(torch.mean(floss))
#         return torch.mean(floss)

# def forward(self, predict, target):
#     pt = F.softmax(predict, dim=1)  # softmmax获取预测概率
#     class_mask = F.one_hot(target, self.class_num)  # 获取target的one hot编码
#     ids = target.view(-1, 1)
#     alpha = self.alpha[ids.data.view(-1)]  # 注意，这里的alpha是给定的一个list(tensor),里面的元素分别是每一个类的权重因子
#     probs = (pt * class_mask).sum(1).view(-1, 1)  # 利用onehot作为mask，提取对应的pt
#     log_p = probs.log()
#     # 同样，原始ce上增加一个动态权重衰减因子
#     loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
#
#     if self.reduction == 'mean':
#         loss = loss.mean()
#     elif self.reduction == 'sum':
#         loss = loss.sum()
#     return loss
