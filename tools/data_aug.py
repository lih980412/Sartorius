import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import json, os
import numpy as np
import PIL.Image as Image
from PIL import ImageDraw
from matplotlib import pyplot as plt
from torchvision import transforms as transforms
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data1(annotation_line, input_shape=None, random=True, hue=.1, sat=1.5, val=1.5, proc_img=True,
                     image_root=None):
    '''random preprocessing for real-time data augmentation'''

    with open(annotation_line, "r") as f:
        p = json.load(f)
        p_images = p["images"]
        p_ann = p["annotations"]

    for i in range(len(p_ann)):
        p_ann[i]["bbox"].append(p_ann[i]["category_id"])
        p_ann[i]["bbox"] = list(map(int, p_ann[i]["bbox"]))

    j = 0
    box = []
    for i in range(0, len(p_images)):
        temp = []
        while j < len(p_ann) and p_ann[j]["image_id"] == p_images[i]["id"]:
            temp.append(p_ann[j]["bbox"])
            temp_np = np.array(temp)
            j += 1
        box.append(temp_np)

    h, w = (p_images[0]["height"], p_images[0]["width"])
    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2
    image_datas = []
    box_datas = []
    index = 0
    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
    place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]

    for i in range(0, 4):
        box[i][:, [2]] = box[i][:, [0]] + box[i][:, [2]]    # (x, y, w, h) ?????? (x, y, x, y)
        box[i][:, [3]] = box[i][:, [1]] + box[i][:, [3]]
        image_name = p_images[i]["file_name"]

        image_path = os.path.join(image_root, image_name)
        image = Image.open(image_path)
        image = image.convert("RGB")
        # image = image.resize()

        image1 = cv2.imread(image_path)
        # # cv2.imshow("image"+str(i), image1)
        # image1 = cv2.rectangle(image1, (p_ann[i]["bbox"][0], p_ann[i]["bbox"][1]),
        #                        (p_ann[i]["bbox"][0] + p_ann[i]["bbox"][2], p_ann[i]["bbox"][1] + p_ann[i]["bbox"][3]),
        #                        (0, 255, 255), 2)
        for j in range(len(box[i])):
            image1 = cv2.rectangle(image1, (box[i][j][0], box[i][j][1]), (box[i][j][2], box[i][j][3]), (0, 255, 255), 2)
        # cv2.imshow("draw" + str(i), image1)

        # ???????????????
        iw, ih = image.size
        # ??????????????????

        # image.save(str(index)+".jpg")
        # ??????????????????
        flip = rand() < .5
        if flip and len(box[i]) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[i][:, [0, 2]] = iw - box[i][:, [2, 0]]

        # ????????????????????????????????????
        new_ar = w / h
        scale = rand(scale_low, scale_high)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ??????????????????
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image = hsv_to_rgb(x)

        # image = np.array(image) / 255.

        image = Image.fromarray((image * 255).astype(np.uint8))
        # ???????????????????????????????????????????????????????????????
        dx = place_x[index]
        dy = place_y[index]
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255
        # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")
        index = index + 1
        box_data = []
        # ???box??????????????????

        if len(box[i]) > 0:
            # np.random.shuffle(box[i])
            box[i][:, [0, 2]] = box[i][:, [0, 2]] * nw / iw + dx
            box[i][:, [1, 3]] = box[i][:, [1, 3]] * nh / ih + dy
            box[i][:, 0:2][box[i][:, 0:2] < 0] = 0
            box[i][:, 2][box[i][:, 2] > w] = w
            box[i][:, 3][box[i][:, 3] > h] = h
            box_w = abs(box[i][:, 2] - box[i][:, 0])
            box_h = abs(box[i][:, 3] - box[i][:, 1])
            box[i] = box[i][np.logical_and(box_w > 1, box_h > 1)]
            box_data = np.zeros((len(box[i]), 5))
            box_data[:len(box[i])] = box[i]
        print(i)
        image_datas.append(image_data)
        box_datas.append(box_data)
        print(box_datas)

        img = Image.fromarray((image_data * 255).astype(np.uint8))
        for j in range(len(box_data)):
            thickness = 3
            left, top, right, bottom = box_data[j][0:4]
            draw = ImageDraw.Draw(img)
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))

        # img.show()

    # ??????????????????????????????
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # ??????????????????????????????
    new_boxes = merge_bboxes(box_datas, cutx, cuty)

    return new_image, new_boxes


def Mosaic(ann_path, image_root):
    mosaic_img, mosaic_ann = get_random_data1(ann_path, image_root=image_root)
    # mosaic_img, mosaic_ann = get_random_data_with_Mosaic(ann_path, image_root)

    # mosaic_img, mosaic_ann = get_random_data(ann_path, (480, 640), image_root=image_root)
    mosaic_img *= 255.
    mosaic_img = mosaic_img.astype(np.uint8)
    print(mosaic_ann)
    mosaic_img = cv2.cvtColor(mosaic_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("mosaic_img", mosaic_img)
    draw_1 = mosaic_img
    for i in range(len(mosaic_ann)):
        draw_1 = cv2.rectangle(draw_1, (int(mosaic_ann[i][0]), int(mosaic_ann[i][1])),
                               (int(mosaic_ann[i][2]), int(mosaic_ann[i][3])),
                               (0, 255, 255), 2)
    return draw_1


def Mixup(image_root):
    alpha = 0.5
    image1 = os.path.join(image_root, "006088.png")
    image2 = os.path.join(image_root, "006089.png")

    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)
    image = (alpha * image1 + (1 - alpha) * image2) / 256.
    return image

    # for model
    # for i, (images, target) in enumerate(train_loader):
    #     # 1.input output
    #     images = images.cuda(non_blocking=True)
    #     target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
    #
    #     # 2.mixup
    #     alpha = config.alpha
    #     lam = np.random.beta(alpha, alpha)
    #     # randperm??????1~images.size(0)?????????????????????
    #     index = torch.randperm(images.size(0)).cuda()
    #     inputs = lam * images + (1 - lam) * images[index, :]
    #     targets_a, targets_b = target, target[index]
    #     outputs = model(inputs)
    #     loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
    #
    #     # 3.backward
    #     optimizer.zero_grad()  # reset gradient
    #     loss.backward()
    #     optimizer.step()  # update parameters of net


def Cutmix(image_batch, alpha=1.0):
    # ??????bbox??????????????????beta??????
    lam = np.random.beta(alpha, alpha)

    #  permutation: ????????????x????????????????????????????????????????????????range(x)
    rand_index = np.random.permutation(len(image_batch))

    # ??????????????????y_a,y_b
    target_a = image_batch_labels
    target_b = image_batch_labels[rand_index]

    # ??????????????????????????????bbox
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_batch[0].shape, lam)

    image_batch_updated = image_batch.copy()

    # image_batch?????????????????? batch x ??? x ??? x ??????
    # ???????????????bbox??????????????? ?????????????????????????????????
    # ???????????????rand_index?????????list?????????????????????list??????????????????image_batch???????????????????????????????????????????????????
    image_batch_updated[:, bbx1: bbx2, bby1:bby2, :] = image_batch[rand_index, bbx1:bbx2, bby1:bby2, :]

    # ?????? 1 - bbox??????????????????????????????
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (image_batch.shape[1] * image_batch.shape[2])
    # ??????????????????label
    label = target_a * lam + target_b * (1. - lam)

    return image_batch_updated, label


if __name__ == "__main__":
    '''PIL'''
    # img = Image.open(r'J:\Data\20210325???????????????????????????\JPEGImages\Threepeople_bar6.jpg').convert("RGB")
    'Resize'
    # new_img = transforms.Resize((100, 200))(img)
    'Crop'
    # new_img = transforms.RandomCrop(300)(img)
    'Rotation'
    # new_img = transforms.RandomRotation(10, expand=True)(img)
    'Flip'
    # new_img = transforms.RandomVerticalFlip(1)(img)
    # new_img = transforms.RandomHorizontalFlip(1)(img)
    # plt.imshow(new_img)
    # plt.show()

    '''cv2'''
    # img = cv2.imread(r'C:\Users\lenovo\Desktop\Li\A-3.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    'Mosaic'
    ann_path = r"J:\Beijing\FSCE_1\datasets\my_dataset\annotations\instances_train.json"
    image_root = r"J:\Beijing\FSCE_1\datasets\my_dataset\image"
    new_img = Mosaic(ann_path, image_root)
    'Mixup'
    # image_root = r"J:\Beijing\FSCE_1\datasets\my_dataset\image"
    # new_img = Mixup(image_root)
    'Cutmix'
    # new_img = Cutmix()

    cv2.imshow("new_img", new_img)
    cv2.waitKey()
