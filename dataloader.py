import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class DataSet(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, img_shape=None):
        self.ann_file = pd.read_excel(annotations_file)
        self.img_dir = img_dir
        self.transform = transform  # 图片扩增
        self.target_transform = target_transform  # 标签处理
        self.img_shape = img_shape

    def __len__(self):
        return len(self.ann_file.id.unique().tolist())

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.ann_file.iloc[idx, 0] + ".png")
        img_input = cv2.imread(img_path)

        mask = np.zeros(self.img_shape)
        labels = self.ann_file[self.ann_file["id"] == self.ann_file.iloc[idx, 0]]["annotation"].tolist()
        for label in labels:
            mask += rle_decode(label, self.img_shape)
        mask = mask.clip(0, 1)
        mask = mask.tolist()

        cell_type = self.ann_file.iloc[idx, 4]

        target = []
        target.append(cell_type)
        target.append(mask)

        if self.transform is not None:
            img_input, target = self.transform(img_input, target)

        return img_input, target

# class DataSet(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, img_shape=None):
#         self.ann_file = pd.read_excel(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform  # 图片
#         self.target_transform = target_transform  # 标签
#         self.img_shape = img_shape
#
#         self.unique_id = self.ann_file.id.unique().tolist()
#
#         img_path = os.path.join(self.img_dir, self.ann_file.iloc[idx, 0] + ".png") for idx in self.unique_id
#
#
#     def __len__(self):
#         return len(self.unique_id)
#
#     def __getitem__(self, idx):
#
#         mask = np.zeros(self.img_shape)
#         labels = self.ann_file[self.ann_file["id"] == self.ann_file.iloc[idx, 0]]["annotation"].tolist()
#         for label in labels:
#             mask += rle_decode(label, self.img_shape)
#         mask = mask.clip(0, 1)
#
#         cell_type = self.ann_file.iloc[idx, 4]
#
#         target = {}
#         target["cell_type"] = cell_type
#         target["mask"] = mask
#
#         if self.transform is not None:
#             img_input, target = self.transform(img_input, target)
#
#         return img_input, target


def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img_de = np.zeros((shape[0] * shape[1]), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img_de[lo: hi] = color
    return img_de.reshape(shape)


# if __name__ == "__main__":
#     annotations_file = "K:\\LiHang\\Cell Instance Segmentation\\train_3.csv"
#     img_root = "K:\\LiHang\\Cell Instance Segmentation\\train"
#     train_set = DataSet(annotations_file, img_root, img_shape=(520, 704))
#
#     train_dataloader = DataLoader(train_set, batch_size=2)
#
#     for index, item in enumerate(train_dataloader):
#         train_features = item[0][index].squeeze()
#         train_labels = item[1]["cell_type"][index]
#         train_mask = item[1]["mask"][index]
#
#     train_features, train_labels = next(iter(train_dataloader))
#     print(f"Feature batch shape: {train_features.size()}")
#     print(f"Labels batch shape: {train_labels}")
#     img = train_features[0].squeeze()
#     label = train_labels["cell_type"]
#     plt.imshow(img)
#     plt.title(label)
#     plt.show()
#     print(f"Label: {label}")