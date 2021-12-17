import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import numpy as np


def vis_features(objectness, layer, batch_id):
    aa = objectness[layer].data.cpu().numpy()[batch_id].transpose(1, 2, 0)
    ff = objectness[layer].data.cpu().numpy()[batch_id].clip(0).transpose(1, 2, 0)
    aa0 = aa[:, :, 0]
    aa[:, :, 0] = (aa0 - np.min(aa0)) / (np.max(aa0) - np.min(aa0) + 0.000001)
    aa1 = aa[:, :, 1]
    aa[:, :, 1] = (aa1 - np.min(aa1)) / (np.max(aa1) - np.min(aa1) + 0.000001)
    aa2 = aa[:, :, 2]
    aa[:, :, 2] = (aa2 - np.min(aa2)) / (np.max(aa2) - np.min(aa2) + 0.000001)

    ff0 = ff[:, :, 0]
    ff[:, :, 0] = (ff0 - np.min(ff0)) / (np.max(ff0) - np.min(ff0) + 0.000001)
    ff1 = ff[:, :, 1]
    ff[:, :, 1] = (ff1 - np.min(ff1)) / (np.max(ff1) - np.min(ff1) + 0.000001)
    ff2 = ff[:, :, 2]
    ff[:, :, 2] = (ff2 - np.min(ff2)) / (np.max(ff2) - np.min(ff2) + 0.000001)

    aa = aa.astype(np.float64)
    ff = ff.astype(np.float64)
    plt.imshow(aa)
    plt.imshow(ff)

    plt.show()

# https://blog.csdn.net/weixin_41735859/article/details/113840457
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch
import torch.nn.functional as F

# ----------------------------------- feature map visualization -----------------------------------

writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")

# 数据加载及预处理
path_img = "./Forsters_Tern_0016_152463.jpg"  # your path to image
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]

norm_transform = transforms.Normalize(normMean, normStd)
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    norm_transform])
img_pil = Image.open(path_img).convert('RGB')
if img_transforms is not None:
    img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)  # chw --> bchw

# 模型加载
vggnet = models.vgg16_bn(pretrained=False)
pthfile = './pretrained/vgg16_bn-6c64b313.pth'
vggnet.load_state_dict(torch.load(pthfile))
# print(vggnet)

# 注册hook
fmap_dict = dict()
n = 0


# for name, sub_module in vggnet.named_modules():  # named_modules()返回网络的子网络层及其名称
#     if isinstance(sub_module, nn.Conv2d):
#         n += 1
#         print('Conv_'+str(n)+'_'+name)

def hook_func(m, i, o):
    # print(m)
    key_name = str(m.weight.shape)
    fmap_dict[key_name].append(o)


for name, sub_module in vggnet.named_modules():  # named_modules()返回网络的子网络层及其名称
    if isinstance(sub_module, nn.Conv2d):
        n += 1
        key_name = str(sub_module.weight.shape)
        # key_name = 'Conv_'+str(n)
        # Python 字典 setdefault() 函数和 get()方法 类似, 如果键不存在于字典中，将会添加键并将值设为默认值。
        fmap_dict.setdefault(key_name, list())
        # print(fmap_dict,'\n')

        n1, n2 = name.split(".")
        # print(n1,n2)
        # print(fmap_dict,'\n')
        # print(name)
        # print('1',vggnet._modules[n1]._modules[n2].named_modules())
        vggnet._modules[n1]._modules[n2].register_forward_hook(hook_func)

# forward
output = vggnet(img_tensor)
print(fmap_dict['torch.Size([128, 64, 3, 3])'][0].shape)
# add image
for layer_name, fmap_list in fmap_dict.items():
    fmap = fmap_list[0]
    # print(fmap.shape)
    fmap.transpose_(0, 1)
    # print(fmap.shape)

    nrow = int(np.sqrt(fmap.shape[0]))
    # if layer_name == 'torch.Size([512, 512, 3, 3])':
    fmap = F.interpolate(fmap, size=[112, 112], mode="bilinear")

    fmap_grid = vutils.make_grid(fmap, normalize=True, scale_each=True, nrow=nrow)
    print(type(fmap_grid), fmap_grid.shape)
    writer.add_image('feature map in {}'.format(layer_name), fmap_grid, global_step=322)