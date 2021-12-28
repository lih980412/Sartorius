import os
import time
import json
import torch
import argparse
import transforms
from coco_utils import evaluate, evaluate_kl
from train_net import create_model
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval

from dataloader import Demo_DataSet, Demo_Defect_DataSet





def parse():
    parser = argparse.ArgumentParser()
    '''data'''
    # for quick
    # parser.add_argument('--train_dir', type=str, default=r"K:\Dataset\MS COCO 2014\coco\train2014", help="train data location")
    parser.add_argument('--train_dir', type=str, default=r"F:\Dataset\MS COCO 2014\coco\train2014",
                        help="train data location")
    parser.add_argument('--val_dir', type=str, default=r"F:\Dataset\MS COCO 2014\coco\val2014",
                        help="val data location")
    parser.add_argument('--ann_dir', type=str, default=r"F:\Dataset\MS COCO 2014\annotations")
    parser.add_argument('--defect_dir', type=str, default=r"D:\UserD\Li\FSCE-1\datasets\my_dataset\image",
                        help="defect data location")
    parser.add_argument('--defect_ann_dir', type=str, default=r"D:\UserD\Li\FSCE-1\datasets\my_dataset\annotations")
    '''hyper-param'''
    parser.add_argument('--weights', type=str, default="", help="model weights")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--num_classes', type=str, default=5, help="the number of dataset's category")
    parser.add_argument('--epoch', type=int, default=2)

    opt = parser.parse_known_args()
    return opt


if __name__ == "__main__":
    cfg = parse()[0]
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # val_file = cfg.val_dir
    defect_file = cfg.defect_dir
    batch_size = cfg.batch_size
    device = torch.device(cfg.device) if torch.cuda.is_available() else "cpu"

    # val_ann = os.path.join(cfg.ann_dir, "instances_val2014.json")
    # val_set = Demo_DataSet(val_file, val_ann, data_transform["val"])
    defect_val_ann = os.path.join(cfg.defect_ann_dir, "instances_val.json")
    val_defect_set = Demo_Defect_DataSet(defect_file, defect_val_ann, data_transform["val"])
    defect_val_dataloader = DataLoader(val_defect_set, batch_size, shuffle=True, pin_memory=True,
                                       collate_fn=Demo_DataSet.collate_fn)

    model = create_model(cfg.num_classes + 1, device)
    model.to(device)


    result = evaluate_kl(model, defect_val_dataloader, device=device, save_dir=save_dir)
    print(result)


