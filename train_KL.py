import time
import torch
import argparse
import transforms
import numpy as np
import os, sys, math
import torch.nn.functional as F
from utils import euclidean_dist
from train_net import create_model
from torch.utils.data import DataLoader, sampler
from coco_utils import warmup_lr_scheduler, evaluate, evaluate_kl
from sklearn.manifold._t_sne import _joint_probabilities
from dataloader import Demo_DataSet, Demo_Defect_DataSet
from FasterRcnn import resnet50_fpn_backbone, FasterRCNN, FastRCNNPredictor, Backbone, GeneralizedRCNNTransform, \
    GeneralizedRCNNTransformKL

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


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
    parser.add_argument('--num_classes', type=str, default=80, help="the number of dataset's category")
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--coco_result_save', type=str, default=r"D:\UserD\Li\Sartorius\predict_result\coco",
                        help="the save directory of the result")
    parser.add_argument('--defect_result_save', type=str, default=r"D:\UserD\Li\Sartorius\predict_result\defect",
                        help="the save directory of the result")
    parser.add_argument('--resume', type=str,
                        default=r"D:\UserD\Li\Sartorius\save_weights\coco\resNetFpn-model-17-0.044.pth",
                        help="the last time checkpoint")

    opt = parser.parse_known_args()
    return opt


def create_KLmodel(min_sizes=800, max_sizes=1333, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d, trainable_layers=5)
    transform = GeneralizedRCNNTransformKL(800, 1333, image_mean, image_std)
    backbone = Backbone(backbone, transform)

    return backbone


def KL_Fea(fea_kl, fea, layers):
    loss_total = 0
    for layer in layers:
        temp = fea[layer].permute(2, 3, 0, 1).reshape(-1, 256).clamp(min=1e-09, max=1e+09)
        temp_kl = fea_kl[layer].permute(2, 3, 0, 1).reshape(-1, 256).clamp(min=1e-09, max=1e+09)
        # P = F.softmax(temp, dim=1)
        # Q = F.log_softmax(temp_kl, dim=1)
        kl_loss = torch.nn.KLDivLoss()
        loss = kl_loss(temp.log(), temp_kl)
        loss_total += loss
    return loss_total


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

    kl_loss = torch.nn.KLDivLoss()
    loss = kl_loss(P, Q)

    # kl_mean = F.kl_div(Q, P, reduction='mean')
    return loss


def train_one_epoch_kl(model, model_kl, optimizer, train_dataloader, defect_dataloader, device, epoch,
                       checkpoint_epoch, warmup=True):
    model.train()
    model_kl.train()
    lr_scheduler = None
    if epoch == 0 and warmup is True and not checkpoint_epoch:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(train_dataloader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    mloss = torch.zeros(1).to(device)  # mean losses
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
        loss_kl = KL_Fea(fea_kl, fea, ["1", "2", "3", "pool"])
        losses = sum(loss for loss in loss_dict.values())

        losses_kl = losses + torch.abs(loss_kl)
        loss_value = losses_kl.item()
        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_value)
            sys.exit(1)
        mloss = (mloss * i + losses_kl.item()) / (i + 1)

        optimizer.zero_grad()
        losses_kl.backward()
        optimizer.step()
        now_lr = optimizer.param_groups[0]["lr"]

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()
        if checkpoint_epoch:
            print("epoch: " + str(epoch + checkpoint_epoch).zfill(4) + ", iter: " + str(i).zfill(
                6) + ", total loss: " + str(
                '%.5f' % loss_value).zfill(7) + ", loss: " + str('%.5f' % losses.item()).zfill(7) + \
                  ", kl_loss: " + str('%.5f' % torch.abs(loss_kl).item()).zfill(7) + ", lr: " + str(
                '%.8f' % now_lr).zfill(
                7))
        else:
            print("epoch: " + str(epoch).zfill(4) + ", iter: " + str(i).zfill(6) + ", total loss: " + str(
                '%.5f' % loss_value).zfill(7) + ", loss: " + str('%.5f' % losses.item()).zfill(7) + \
                  ", kl_loss: " + str('%.5f' % torch.abs(loss_kl).item()).zfill(7) + ", lr: " + str(
                '%.8f' % now_lr).zfill(
                7))

    return mloss, now_lr


def main(cfg):
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor()]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    '''DataSet'''
    train_file = cfg.train_dir
    train_ann = os.path.join(cfg.ann_dir, "instances_train2014.json")
    train_set = Demo_DataSet(train_file, train_ann, data_transform["train"])
    val_file = cfg.val_dir
    val_ann = os.path.join(cfg.ann_dir, "instances_minival2014.json")
    val_set = Demo_DataSet(val_file, val_ann, data_transform["val"])

    defect_file = cfg.defect_dir
    defect_ann = os.path.join(cfg.defect_ann_dir, "instances_train.json")
    train_defect_set = Demo_Defect_DataSet(defect_file, defect_ann, data_transform["train"])
    # defect_val_ann = os.path.join(cfg.defect_ann_dir, "instances_val.json")
    # val_defect_set = Demo_Defect_DataSet(defect_file, defect_val_ann, data_transform["val"])

    '''Dataloader'''
    batch_size = cfg.batch_size
    # train_sampler = sampler.RandomSampler(data_source=train_set, replacement=True)
    train_dataloader = DataLoader(train_set, batch_size, shuffle=True,
                                  collate_fn=Demo_DataSet.collate_fn, pin_memory=True)
    # defect_sampler = sampler.RandomSampler(data_source=train_defect_set, replacement=True)
    defect_dataloader = DataLoader(train_defect_set, batch_size, shuffle=True,
                                   collate_fn=Demo_DataSet.collate_fn, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size, shuffle=True, pin_memory=True, collate_fn=Demo_DataSet.collate_fn)
    # defect_val_dataloader = DataLoader(val_defect_set, batch_size, shuffle=True, pin_memory=True,
    #                                    collate_fn=Demo_DataSet.collate_fn)

    '''Model'''
    device = torch.device(cfg.device) if torch.cuda.is_available() else "cpu"
    model = create_model(cfg.num_classes + 1, device)
    model.to(device)
    model_kl = create_KLmodel(800, 1333, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    model_kl.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    params_kl = [p for p in model_kl.parameters() if p.requires_grad]
    params += params_kl
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)

    # params_kl = [p for p in model_kl.parameters() if p.requires_grad]
    # optimizer_kl = torch.optim.SGD(params_kl, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # lr_scheduler_kl = torch.optim.lr_scheduler.MultiStepLR(optimizer_kl, milestones=[16, 22], gamma=0.1)

    train_loss, kl_loss = [], []
    train_lr, kl_lr = [], []
    best_val = 0.
    # write into txt
    file_name = os.path.join(cfg.coco_result_save, "result_" + time.strftime('%m%d%H%M%S', time.localtime()) + ".txt")
    file_name_kl = os.path.join(cfg.defect_result_save,
                                "result_" + time.strftime('%m%d%H%M%S', time.localtime()) + ".txt")

    if cfg.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(cfg.resume, map_location=device)  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.start_epoch = checkpoint['epoch'] + 1
        checkpoint_epoch = checkpoint['epoch'] + 1
        print("continue train model at epoch " + str(checkpoint['epoch'] + 1))

    for epoch in range(cfg.epoch):
        loss, lr = train_one_epoch_kl(model, model_kl, optimizer, train_dataloader, defect_dataloader, device, epoch,
                                      checkpoint_epoch)
        train_loss.append(loss.item())
        train_lr.append(lr)

        lr_scheduler.step()

        '''val CoCo dataset'''
        print("----------------------evaluate--------------------------------")
        coco_info = evaluate(model, val_dataloader, device=device, save_dir=cfg.coco_result_save, epoch=epoch,
                             checkpoint_epoch=checkpoint_epoch)
        print(coco_info[0])
        print("--------------------------------------------------------------")

        with open(file_name, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [str(round(i, 4)) for i in coco_info + [loss.item()]] + [str(round(lr, 6))]
            if cfg.resume:
                txt = "epoch:{} {}".format(epoch + checkpoint_epoch, '  '.join(result_info))
            else:
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        if round(coco_info[0], 4) > best_val:
            print("save best val " + str(coco_info[0]) + ".pth to ./save_weights/coco")
            # save weights
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            if cfg.resume:
                torch.save(save_files,
                           "./save_weights/coco/resNetFpn-model-{}-{}.pth".format(epoch + checkpoint_epoch,
                                                                                  round(coco_info[0], 4)))
                best_val = round(coco_info[0], 4)
            else:
                torch.save(save_files,
                           "./save_weights/coco/resNetFpn-model-{}-{}.pth".format(epoch, round(coco_info[0], 4)))
                best_val = round(coco_info[0], 4)

        '''val Defect dataset'''
        # print("----------------------evaluate--------------------------------")
        # coco_info = evaluate_kl(model, defect_val_dataloader, device=device, save_dir=cfg.defect_result_save, epoch=epoch)
        # print(coco_info[0])
        # print("--------------------------------------------------------------")
        #
        #
        # with open(file_name_kl, "a") as f:
        #     # 写入的数据包括coco指标还有loss和learning rate
        #     result_info = [str(round(i, 4)) for i in coco_info + [loss.item()]] + [str(round(lr, 6))]
        #     txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        #     f.write(txt + "\n")
        #
        # if round(coco_info[0], 4) > best_val:
        #     print("save best val " + str(coco_info[0]) + ".pth to ./save_weights/defect")
        #     # save weights
        #     save_files = {
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch}
        #     torch.save(save_files,
        #                "./save_weights/defect/resNetFpn-model-{}-{}.pth".format(epoch).format(round(coco_info[0], 4)))
        #     best_val = round(coco_info[0], 4)


if __name__ == "__main__":
    cfg = parse()[0]
    # main_kl(cfg)
    main(cfg)
