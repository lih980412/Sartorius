import os
import cv2
import json
import torch
import itertools
import numpy as np
import pandas as pd
from PIL import Image
# from PIL import ImageFile
from pycocotools.coco import COCO
from torch.utils.data import Dataset
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from coco_utils import COCO_CATEGORIES, DEFECT_CATEGORIES



class Demo_Defect_DataSet(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        super(Demo_Defect_DataSet, self).__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.coco = COCO(annotations_file)

        thing_ids = [k["id"] for k in DEFECT_CATEGORIES if k["isthing"] == 1]
        assert len(thing_ids) == 5, len(thing_ids)
        self.thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}

        ids = list(sorted(self.coco.imgs.keys()))
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):  # 这里的 idx 是这个 epoch 内图片的序号
        coco = self.coco
        img_id = self.ids[idx]
        img_name = coco.loadImgs([img_id])[0]["file_name"]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        ann_id = coco.getAnnIds(imgIds=[img_id])
        img_ann = [obj for obj in coco.loadAnns(ann_id) if obj["iscrowd"] == 0]
        target = {}
        boxes = []
        for obj in img_ann:
            if obj["bbox"][2] > 0 and obj["bbox"][3] > 0:
                boxes.append(obj["bbox"])
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        # 这两句与上面一句等价
        # boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        # boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        w, h = img.size
        if (w is not None) and (h is not None):
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
        target["boxes"] = boxes

        classes = [self.thing_dataset_id_to_contiguous_id[obj["category_id"]] for obj in img_ann]
        classes = torch.as_tensor(classes, dtype=torch.int64)
        target["labels"] = classes
        target["image_id"] = torch.as_tensor([int(img_id)])
        target["area"] = torch.as_tensor([obj["area"] for obj in img_ann])
        target["iscrowd"] = torch.as_tensor([obj["iscrowd"] for obj in img_ann])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class Demo_DataSet(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        super(Demo_DataSet, self).__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.coco = COCO(annotations_file)

        thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
        # thing_ornot_ids = [k["id"] for k in COCO_CATEGORIES]
        assert len(thing_ids) == 80, len(thing_ids)
        self.thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
        ids = list(sorted(self.coco.imgs.keys()))
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):  # 这里的 idx 是这个 epoch 内图片的序号
        coco = self.coco
        img_id = self.ids[idx]
        img_name = coco.loadImgs(img_id)[0]["file_name"]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        ann_id = coco.getAnnIds(imgIds=img_id)
        img_ann = [obj for obj in coco.loadAnns(ann_id) if obj["iscrowd"] == 0]
        target = {}
        boxes = []
        for obj in img_ann:
            if obj["bbox"][2] > 0 and obj["bbox"][3] > 0:
                boxes.append(obj["bbox"])
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        # 这两句与上面一句等价
        # boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        # boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        w, h = img.size
        if (w is not None) and (h is not None):
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
        target["boxes"] = boxes

        classes = [self.thing_dataset_id_to_contiguous_id[obj["category_id"]] for obj in img_ann]
        classes = torch.as_tensor(classes, dtype=torch.int64)
        target["labels"] = classes
        target["image_id"] = torch.as_tensor([img_id])
        target["area"] = torch.as_tensor([obj["area"] for obj in img_ann])
        target["iscrowd"] = torch.as_tensor([obj["iscrowd"] for obj in img_ann])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))




class Faster_DataSet(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, img_shape=None):
        super(Faster_DataSet, self).__init__()
        self.img_dir = img_dir
        self.transform = transform  # 图片扩增
        # self.target_transform = target_transform  # 标签处理
        self.coco = COCO(annotations_file)

        thing_ids = [k["id"] for k in COCO_CATEGORIES if k["isthing"] == 1]
        assert len(thing_ids) == 80, len(thing_ids)
        self.thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
        # thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
        ids = list(sorted(self.coco.imgs.keys()))
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        w, h = img.size
        anno = [obj for obj in coco_target if obj['iscrowd'] == 0]
        boxes = []
        for obj in anno:
            if obj["bbox"][2] > 0 and obj["bbox"][3] > 0:
                boxes.append(obj["bbox"])
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        if (w is not None) and (h is not None):
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [self.thing_dataset_id_to_contiguous_id[obj["category_id"]] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = torch.tensor([img_id])
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class Mask_DataSet(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, img_shape=None):
        super(Mask_DataSet, self).__init__()
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
