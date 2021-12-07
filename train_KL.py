import torch
import argparse
import transforms
import numpy as np
import os, sys, math
import torch.nn.functional as F
from utils import euclidean_dist
from train_net import create_model
from torch.utils.data import DataLoader
from coco_utils import warmup_lr_scheduler, evaluate
from sklearn.manifold._t_sne import _joint_probabilities
from dataloader import Demo_DataSet, Demo_Defect_DataSet
from FasterRcnn import resnet50_fpn_backbone, FasterRCNN, FastRCNNPredictor, Backbone, GeneralizedRCNNTransform, \
    GeneralizedRCNNTransformKL


def parse():
    parser = argparse.ArgumentParser()
    '''data'''
    # for quick
    # parser.add_argument('--train_dir', type=str, default=r"K:\Dataset\MS COCO 2014\coco\train2014", help="train data location")
    parser.add_argument('--train_dir', type=str, default=r"K:\Dataset\MS COCO 2014\coco\train2014",
                        help="train data location")
    parser.add_argument('--val_dir', type=str, default=r"K:\Dataset\MS COCO 2014\coco\val2014",
                        help="val data location")
    parser.add_argument('--ann_dir', type=str, default=r"K:\Dataset\MS COCO 2014")
    parser.add_argument('--defect_dir', type=str, default=r"J:\Beijing\FSCE-1\datasets\my_dataset\image",
                        help="defect data location")
    parser.add_argument('--defect_ann_dir', type=str, default=r"J:\Beijing\FSCE-1\datasets\my_dataset\annotations")
    '''hyper-param'''
    parser.add_argument('--weights', type=str, default="", help="model weights")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--num_classes', type=str, default=80, help="the number of dataset's category")
    parser.add_argument('--epoch', type=int, default=2)

    opt = parser.parse_known_args()
    return opt


def create_KLmodel(min_sizes=800, max_sizes=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d, trainable_layers=3)
    transform = GeneralizedRCNNTransformKL(800, 1333, image_mean, image_std)
    backbone = Backbone(backbone, transform)

    return backbone


def KL(fea_kl, fea):
    temp = fea["2"][0].permute(1, 2, 0).reshape(-1, 256)
    temp_kl = fea_kl["2"][0].permute(1, 2, 0).reshape(-1, 256)
    '''计算各样本间距离，特征图上的每点作为一个样本，第三个通道的256维作为样本特征'''
    distance = euclidean_dist(temp, temp)
    distance_kl = euclidean_dist(temp_kl, temp_kl)
    '''计算联合分布'''
    distance_np = np.asarray(distance.detach().cpu())
    P = _joint_probabilities(distances=distance_np, desired_perplexity=25., verbose=False)
    P = torch.tensor(P, dtype=torch.float32, device="cuda")
    distance_kl_np = np.asarray(distance_kl.detach().cpu())
    Q = _joint_probabilities(distances=distance_kl_np, desired_perplexity=25., verbose=False)
    Q = torch.tensor(Q, dtype=torch.float32, device="cuda")
    '''归一化'''
    P = F.softmax(P, dim=0)
    Q = F.log_softmax(Q, dim=0)
    '''计算损失'''
    kl_mean = F.kl_div(Q, P, reduction='mean')
    return kl_mean

# def main_kl(cfg):
#     data_transform = {
#         "train": transforms.Compose([transforms.ToTensor()]),
#         "val": transforms.Compose([transforms.ToTensor])
#     }
#     train_file = cfg.train_dir
#     train_ann = os.path.join(cfg.ann_dir, "instances_minival2014.json")
#     train_set = Demo_DataSet(train_file, train_ann, data_transform["train"])
#     # val_file = os.path.join(cfg.val_dir)
#     # val_ann = os.path.join(cfg.ann_dir, "instances_val2014.json")
#     # val_set = Demo_DataSet(val_file, val_ann, data_transform)
#
#     defect_file = cfg.defect_dir
#     defect_ann = os.path.join(cfg.defect_ann_dir, "instances_val.json")
#     train_defect_set = Demo_Defect_DataSet(defect_file, defect_ann, data_transform["train"])
#
#     batch_size = cfg.batch_size
#     train_dataloader = DataLoader(train_set,
#                                   batch_size,
#                                   shuffle=True,
#                                   collate_fn=Demo_DataSet.collate_fn,
#                                   pin_memory=True)
#     defect_dataloader = DataLoader(train_defect_set,
#                                    batch_size,
#                                    shuffle=True,
#                                    collate_fn=Demo_DataSet.collate_fn,
#                                    pin_memory=True)
#     device = torch.device(cfg.device) if torch.cuda.is_available() else "cpu"
#
#     model_kl = create_KLmodel(cfg.num_classes + 1, device)
#     model_kl.to(device)
#     params_kl = [p for p in model_kl.parameters() if p.requires_grad]
#     optimizer_kl = torch.optim.SGD(params_kl, lr=0.005, momentum=0.9, weight_decay=0.0005)
#     lr_scheduler_kl = torch.optim.lr_scheduler.MultiStepLR(optimizer_kl, milestones=[16, 22], gamma=0.1)
#
#     train_loss = []
#     kl_loss = []
#     # learning_rate = []
#     # val_map = []
#
#     for epoch in range(cfg.epoch):
#         # loss = train_one_epoch(model, model_kl, optimizer, optimizer_kl, train_dataloader, defect_dataloader, device, epoch)
#
#         loss = train_one_epoch_kl(model_kl, optimizer_kl, train_dataloader, defect_dataloader, device, epoch)
#
#
# def train_one_epoch_kl(model_kl, optimizer_kl, train_dataloader, defect_dataloader, device, epoch, warmup=True):
#     model_kl.train()
#     if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
#         warmup_factor = 1.0 / 1000
#         warmup_iters = min(1000, len(defect_dataloader) - 1)
#         lr_scheduler_kl = warmup_lr_scheduler(optimizer_kl, warmup_iters, warmup_factor)
#     for i, ([images, targets], [def_images, def_targets]) in enumerate(zip(train_dataloader, defect_dataloader)):
#         shapes, imageS, def_imgs = [], [], []
#         for image in images:
#             image = image.to(device)
#             shapes.append([image.size()])  # (C, H, W)
#             imageS.append(image)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         def_images = list(image.to(device) for image in def_images)
#         deF_images = []
#         for (image, shape) in zip(def_images, shapes):
#             resize_transform = transforms.Resize(image.shape[1], image.shape[2], shape)
#             deF_images.append(torch.as_tensor(resize_transform.apply_image(image), device="cuda"))
#         del def_images
#         # feature map
#         fea_kl = model_kl(deF_images)
#         temp = fea_kl["2"][0].permute(1, 2, 0).reshape(-1, 256)
#         # 计算各样本间距离，特征图上的每点作为一个样本，第三个通道的256维作为样本特征
#         distance = euclidean_dist(temp, temp)
#         # 计算联合分布
#         distance_np = np.asarray(distance.detach().cpu())
#         P = _joint_probabilities(distances=distance_np, desired_perplexity=25., verbose=False)
#         P = torch.tensor(P, dtype=torch.float32, device="cuda")
#         # 归一化
#         P = F.softmax(P, dim=0)
#         kl_mean = F.kl_div(log_a, P, reduction='mean')
#         print(fea_kl)
#         # fea, loss_dict = model(imageS, targets)


def train_one_epoch(model, model_kl, optimizer, optimizer_kl, train_dataloader, defect_dataloader, device, epoch,
                    warmup=True):
    model.train()
    model_kl.train()
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(train_dataloader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        lr_scheduler_kl = warmup_lr_scheduler(optimizer_kl, warmup_iters, warmup_factor)
    mloss = torch.zeros(1).to(device)  # mean losses
    mloss_kl = torch.zeros(1).to(device)
    for i, ([images, targets], [def_images, def_targets]) in enumerate(zip(train_dataloader, defect_dataloader)):
        shapes, imageS, def_imgs = [], [], []
        for image in images:
            image = image.to(device)
            shapes.append([image.size()])  # (C, H, W)
            imageS.append(image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        def_images = list(image.to(device) for image in def_images)
        deF_images = []
        # targets = [{k: v.to(device) for k, v in t.items()} for t in def_targets]
        for (image, shape) in zip(def_images, shapes):
            resize_transform = transforms.Resize(image.shape[1], image.shape[2], shape)
            deF_images.append(torch.as_tensor(resize_transform.apply_image(image), device="cuda"))
        del def_images, images

        # feature map
        fea_kl = model_kl(deF_images)
        fea, loss_dict = model(imageS, targets)
        loss_kl = KL(fea_kl, fea)
        losses = sum(loss for loss in loss_dict.values())

        losses_kl = losses + loss_kl
        loss_value = losses_kl.item()
        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_value)
            sys.exit(1)
        mloss = (mloss * i + losses.item()) / (i + 1)
        mloss_kl = (mloss_kl * i + loss_kl.item()) / (i + 1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        now_lr = optimizer.param_groups[0]["lr"]

        optimizer_kl.zero_grad()
        loss_kl.backward()
        optimizer_kl.step()
        now_lr_kl = optimizer.param_groups[0]["lr"]

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()
        print("epoch: " + str(epoch).zfill(4) + ", iter: " + str(i).zfill(6) + ", total loss: " + str('%.3f' % loss_value).zfill(7)+ ", loss: " + str('%.3f' % losses.item()).zfill(7) + \
              ", kl_loss: " + str('%.3f' % loss_kl.item()).zfill(7) + ", lr: " + str('%.8f' % now_lr).zfill(7) + ", kl_lr: " + str('%.8f' % now_lr_kl).zfill(7))

    return mloss, mloss_kl, now_lr, now_lr_kl


def main(cfg):
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor])
    }
    train_file = cfg.train_dir
    train_ann = os.path.join(cfg.ann_dir, "instances_minival2014.json")
    train_set = Demo_DataSet(train_file, train_ann, data_transform["train"])
    val_file = cfg.val_dir
    val_ann = os.path.join(cfg.ann_dir, "instances_val2014.json")
    val_set = Demo_DataSet(val_file, val_ann, data_transform["val"])
    defect_file = cfg.defect_dir
    defect_ann = os.path.join(cfg.defect_ann_dir, "instances_val.json")
    train_defect_set = Demo_Defect_DataSet(defect_file, defect_ann, data_transform["train"])

    batch_size = cfg.batch_size
    train_dataloader = DataLoader(train_set,
                                  batch_size,
                                  shuffle=True,
                                  collate_fn=Demo_DataSet.collate_fn,
                                  pin_memory=True)
    defect_dataloader = DataLoader(train_defect_set,
                                   batch_size,
                                   shuffle=True,
                                   collate_fn=Demo_DataSet.collate_fn,
                                   pin_memory=True)

    val_dataloader = DataLoader(val_set,
                                batch_size,
                                shuffle=True,
                                pin_memory=True,
                                collate_fn=Demo_DataSet.collate_fn)
    device = torch.device(cfg.device) if torch.cuda.is_available() else "cpu"

    model = create_model(cfg.num_classes + 1, device)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)

    model_kl = create_KLmodel(cfg.num_classes + 1, device)
    model_kl.to(device)
    params_kl = [p for p in model_kl.parameters() if p.requires_grad]
    optimizer_kl = torch.optim.SGD(params_kl, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler_kl = torch.optim.lr_scheduler.MultiStepLR(optimizer_kl, milestones=[16, 22], gamma=0.1)

    train_loss, kl_loss = [], []
    train_lr, kl_lr = [], []

    # val_map = []

    for epoch in range(cfg.epoch):
        loss, loss_kl, lr, lr_kl = train_one_epoch(model, model_kl, optimizer, optimizer_kl, train_dataloader, defect_dataloader, device, epoch)
        train_loss.append(loss.item())
        kl_loss.append(loss_kl.item())
        train_lr.append(lr)
        kl_lr.append(lr_kl)

        lr_scheduler.step()
        lr_scheduler_kl.step()

        # evaluate on the test dataset
        coco_info = evaluate(model, val_dataloader, device=device)
        print("evaluate--------------------------------")
        print(coco_info[1])
        print("----------------------------------------")
        # # write into txt
        # with open(results_file, "a") as f:
        #     # 写入的数据包括coco指标还有loss和learning rate
        #     result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
        #     txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        #     f.write(txt + "\n")
        #
        # val_map.append(coco_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, "./save_weights/resNetFpn-model-{}.pth".format(epoch))


if __name__ == "__main__":
    cfg = parse()[0]
    # main_kl(cfg)
    main(cfg)
