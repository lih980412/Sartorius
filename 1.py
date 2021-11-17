# import torch
# from torch._six import int_classes as _int_classes
#
#
# class Sampler(object):
#     r"""Base class for all Samplers.
#     Every Sampler subclass has to provide an __iter__ method, providing a way
#     to iterate over indices of dataset elements, and a __len__ method that
#     returns the length of the returned iterators.
#     """
#
#     def __init__(self, data_source):
#         pass
#
#     def __iter__(self):
#         raise NotImplementedError
#
#     def __len__(self):
#         raise NotImplementedError
#
#
# class SequentialSampler(Sampler):
#     r"""Samples elements sequentially, always in the same order.
#     Arguments:
#         data_source (Dataset): dataset to sample from
#     """
#
#     def __init__(self, data_source):
#         self.data_source = data_source
#
#     def __iter__(self):
#         return iter(range(len(self.data_source)))
#
#     def __len__(self):
#         return len(self.data_source)
#
#
# class RandomSampler(Sampler):
#     r"""Samples elements randomly, without replacement.
#     Arguments:
#         data_source (Dataset): dataset to sample from
#     """
#
#     def __init__(self, data_source):
#         self.data_source = data_source
#
#     def __iter__(self):
#         return iter(torch.randperm(len(self.data_source)).tolist())
#
#     def __len__(self):
#         return len(self.data_source)
#
#
# class BatchSampler(Sampler):
#     r"""Wraps another sampler to yield a mini-batch of indices.
#     Args:
#         sampler (Sampler): Base sampler.
#         batch_size (int): Size of mini-batch.
#         drop_last (bool): If ``True``, the sampler will drop the last batch if
#             its size would be less than ``batch_size``
#     Example:
#         >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
#         [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
#         >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
#         [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
#     """
#
#     def __init__(self, sampler, batch_size, drop_last):
#         if not isinstance(sampler, Sampler):
#             raise ValueError("sampler should be an instance of "
#                              "torch.utils.data.Sampler, but got sampler={}"
#                              .format(sampler))
#         if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
#                 batch_size <= 0:
#             raise ValueError("batch_size should be a positive integeral value, "
#                              "but got batch_size={}".format(batch_size))
#         if not isinstance(drop_last, bool):
#             raise ValueError("drop_last should be a boolean value, but got "
#                              "drop_last={}".format(drop_last))
#         self.sampler = sampler
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#
#     def __iter__(self):
#         batch = []
#         # 一旦达到batch_size的长度，说明batch被填满，就可以yield出去了
#         for idx in self.sampler:
#             batch.append(idx)
#             if len(batch) == self.batch_size:
#                 yield batch
#                 batch = []
#         if len(batch) > 0 and not self.drop_last:
#             yield batch
#
#     def __len__(self):
#         # 比如epoch有100个样本，batch_size选择为64，那么drop_last的结果为1，不drop_last的结果为2
#         if self.drop_last:
#             return len(self.sampler) // self.batch_size
#         else:
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size
#
#
# if __name__ == "__main__":
#     # random = list(RandomSampler(range(10)))
#     # print(random)
#     # print(list(SequentialSampler(range(10))))
#     # for idx in random:
#     #     print(idx)
#
#     Batch = BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False)
#     batch = iter(Batch)
#     # for data in batch:
#     #     print(data)
#     print(next(batch))
#     print(next(batch))
#     print(next(batch))
#     print(next(batch))
#
# # def MyRange(end):
# #     start = 0
# #     while start < end:
# #         x = yield start  # 这里增加了获取返回值
# #         print(x)  # 打印出来
# #         start += 1
# #
# # m = MyRange(5)
# # print(next(m))
# # print(m.send(10))

'''
Focal loss
'''
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


# https://www.cnblogs.com/Henry-ZHAO/p/13087275.html
# https://blog.csdn.net/wuliBob/article/details/104119616
def cross_entropy_error(class_num, predict, target):
    delta = 1e-7
    normal = len(target)
    predict = F.softmax(predict, dim=1)
    '''1'''
    # target = F.one_hot(target, class_num)
    # return format((-np.sum((target * np.log(predict + delta)).numpy()) / normal), '.4f')
    # print(-(target * np.log(predict + delta)).numpy().sum(axis=1))
    '''2'''
    return F.nll_loss(np.log(predict + delta), target)


def focal_loss(class_num, predict, target):
    alpha = Variable(torch.ones(class_num, 1))
    gamma = 2
    reduction = 'mean'
    pt = F.softmax(predict, dim=1)  # softmmax获取预测概率
    class_mask = F.one_hot(target, class_num)  # 获取target的one hot编码
    ids = target.view(-1, 1)
    alpha = alpha[ids.data.view(-1)].view(-1, 1)  # 注意，这里的alpha是给定的一个list(tensor),里面的元素分别是每一个类的权重因子
    probs = (pt * class_mask).sum(1).view(-1, 1)  # 利用onehot作为mask，提取对应的pt
    log_p = probs.log()
    # 同样，原始ce上增加一个动态权重衰减因子
    loss = -alpha * (torch.pow((1 - probs), gamma)) * log_p
    # print(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss

# class FocalLoss(nn.Module):
#     # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
#     def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
#         super(FocalLoss, self).__init__()
#         self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = loss_fcn.reduction
#         self.loss_fcn.reduction = 'none'  # required to apply FL to each element
#
#     def forward(self, pred, true):
#         loss = self.loss_fcn(pred, true)
#         # p_t = torch.exp(-loss)
#         # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability
#
#         # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
#         pred_prob = torch.sigmoid(pred)  # prob from logits
#         p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
#         alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
#         modulating_factor = (1.0 - p_t) ** self.gamma
#         loss *= alpha_factor * modulating_factor
#
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:  # 'none'
#             return loss




def focal_loss1(class_num, predict, target):
    eps = 1e-7
    weight = 0.25
    gamma = 2

    class_mask = F.one_hot(target, class_num)
    # y_pred = predict.view(predict.size()[0], predict.size()[1])  # B*C*H*W->B*C*(H*W)
    y_pred = F.softmax(predict, dim=1)

    target = class_mask.view(y_pred.size())  # B*C*H*W->B*C*(H*W)
    ce = -1 * torch.log(y_pred + eps) * target
    floss = torch.pow((1 - y_pred), gamma) * ce
    floss = torch.mul(floss, weight)

    floss = torch.sum(floss, dim=1)
    # print(floss)
    # print(torch.mean(floss))
    return torch.mean(floss)


if __name__ == "__main__":
    predict = torch.tensor([[1.2171, 1.5767, 0.9030, 0.8844, -0.5551],
                            [-0.6737, -0.4417, -1.1523, -0.9431, 0.1386],
                            [-1.0912, 0.3444, -0.9349, -0.1763, -0.7261]])
    predict = torch.tensor([[0.1, 0.1, 0.8, 0.],  # p-easy
                            [0.2, 0.2, 0.31, 0.29],  # p-hard
                            [0.32, 0.28, 0.3101, 0.2899]])  # n
    target = torch.tensor([2, 2, 0])  # 一个是第4类，一个是第3类，一个是第1类
    # target1 = torch.tensor([[0, 0, 1], [0, 0, 1]])
    class_num = 4
    # print("focal loss: " + str(focal_loss(class_num, predict, target)))
    print("focal loss1: " + str(focal_loss1(class_num, predict, target)))
    print("cross entropy: " + str(cross_entropy_error(class_num, predict, target)))
    print(F.cross_entropy(predict, target))
