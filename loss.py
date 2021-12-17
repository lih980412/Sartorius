import torch
import random
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

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul: 多核MMD，以bandwidth为中心，两边扩展的基数，比如bandwidth/kernel_mul, bandwidth, bandwidth*kernel_mul
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值，是否固定，如果固定，则为单核MMD
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0]) + int(target.size()[0])  # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)  # 将source,target按列方向合并
    # 将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    # 求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    # total1 - total2 得到的矩阵中坐标（i,j, :）代表total中第i行数据和第j行数据之间的差
    # sum函数，对第三维进行求和，即平方后再求和，获得高斯核指数部分的分子，是L2范数的平方
    L2_distance = ((total0 - total1) ** 2).sum(2)
    # 调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    # 多核MMD
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    # 得到最终的核矩阵
    return sum(kernel_val)
    # return sum(kernel_val)/len(kennel_val)


def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
        计算源域数据和目标域数据的MMD距离
        Params:
    	    source: 源域数据（n * len(x))
    	    target: 目标域数据（m * len(y))
    	    kernel_mul:
    	    kernel_num: 取不同高斯核的数量
    	    fix_sigma: 不同高斯核的sigma值
    	Return:
    		loss: MMD loss
        '''
    batch_size = int(source.size()[0])  # 一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)

    return loss  # 因为一般都是n==m，所以L矩阵一般不加入计算


if __name__ == "__main__":
    SAMPLE_SIZE = 500

    # 分别从对数正态分布和beta分布取两组数据
    diff_1 = []
    mu = -0.6
    sigma = 0.15  # 将输出数据限制到0-1之间
    for i in range(10):
        diff_1.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])

    # diff_2 = []
    # alpha = 1
    # beta = 10
    # for i in range(10):
    #     diff_2.append([random.betavariate(alpha, beta) for _ in range(1, SAMPLE_SIZE)])

    same_2 = []
    for i in range(10):
        same_2.append([random.lognormvariate(mu, sigma) for _ in range(1, SAMPLE_SIZE)])

    X = torch.Tensor(diff_1)
    # Y = torch.Tensor(diff_2)
    Y = torch.Tensor(same_2)
    X, Y = Variable(X), Variable(Y)
    print(MMD(X, Y))