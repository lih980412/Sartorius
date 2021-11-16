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

# 验证 Focal loss
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def cross_entropy_error(predict, target):
    delta = 1e-7
    print(target * np.log(predict))
    # return -np.sum(t * np.log(y + delta))
    # print(-np.sum(t[0] * np.log(y[0] + delta)))
    # print(-np.sum(t[1] * np.log(y[1] + delta)))

def focal_loss(class_num, predict, target):
    alpha = Variable(torch.ones(class_num, 1))
    gamma = 2
    reduction = 'mean'
    pt = F.softmax(predict, dim=1)  # softmmax获取预测概率
    class_mask = F.one_hot(target, class_num)  # 获取target的one hot编码
    ids = target.view(-1, 1)
    alpha = alpha[ids.data.view(-1)]  # 注意，这里的alpha是给定的一个list(tensor),里面的元素分别是每一个类的权重因子
    probs = (pt * class_mask).sum(1).view(-1, 1)  # 利用onehot作为mask，提取对应的pt
    log_p = probs.log()
    # 同样，原始ce上增加一个动态权重衰减因子
    loss = -alpha * (torch.pow((1 - probs), gamma)) * log_p
    print(loss)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    # print(loss)


if __name__ == "__main__":
    predict = torch.tensor([[0.06, 0.03, 0.91], [0.08, 0.02, 0.9]])
    target = torch.tensor([2, 2])
    target1 = torch.tensor([[0, 0, 1], [0, 0, 1]])
    class_num = 3
    focal_loss(class_num, predict, target)
    cross_entropy_error(predict, target1)
