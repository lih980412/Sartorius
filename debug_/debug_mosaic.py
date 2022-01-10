from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import ImageDraw
from PIL import Image
import numpy as np
import json, os
import cv2

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a
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
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


'''not work'''
def get_random_data_with_Mosaic(annotation_line, image_root, input_shape=None, max_boxes=100, hue=.1, sat=1.5, val=1.5):
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

    h, w = p_images[0]["height"], p_images[0]["width"]
    min_offset_x = rand(0.25, 0.75)
    min_offset_y = rand(0.25, 0.75)

    nws = [int(w * rand(0.4, 1)), int(w * rand(0.4, 1)), int(w * rand(0.4, 1)),
           int(w * rand(0.4, 1))]
    nhs = [int(h * rand(0.4, 1)), int(h * rand(0.4, 1)), int(h * rand(0.4, 1)),
           int(h * rand(0.4, 1))]

    place_x = [int(w * min_offset_x) - nws[0], int(w * min_offset_x) - nws[1], int(w * min_offset_x),
               int(w * min_offset_x)]
    place_y = [int(h * min_offset_y) - nhs[0], int(h * min_offset_y), int(h * min_offset_y),
               int(h * min_offset_y) - nhs[3]]

    image_datas = []
    box_datas = []
    index = 0
    for i in range(0, 4):
        image_name = p_images[i]["file_name"]

        image_path = os.path.join(image_root, image_name)
        image = Image.open(image_path)
        image = image.convert("RGB")
        # image = image.resize()

        image1 = cv2.imread(image_path)
        # cv2.imshow("image"+str(i), image1)
        image1 = cv2.rectangle(image1, (p_ann[i]["bbox"][0], p_ann[i]["bbox"][1]),
                               (p_ann[i]["bbox"][0] + p_ann[i]["bbox"][2], p_ann[i]["bbox"][1] + p_ann[i]["bbox"][3]),
                               (0, 255, 255), 2)
        cv2.imshow("draw" + str(i), image1)

        # 图片的大小
        iw, ih = image.size
        # 保存框的位置

        # 是否翻转图片
        flip = rand() < .5
        if flip and len(box[i]) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[i][:, [0, 2]] = iw - box[i][:, [2, 0]]

        nw = nws[index]
        nh = nhs[index]
        image = image.resize((nw, nh), Image.BICUBIC)

        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image)

        index = index + 1
        box_data = []
        # 对box进行重新处理
        if len(box[i]) > 0:
            np.random.shuffle(box[i])
            box[i][:, [0, 2]] = box[i][:, [0, 2]] * nw / iw + dx
            box[i][:, [1, 3]] = box[i][:, [1, 3]] * nh / ih + dy
            box[i][:, 0:2][box[i][:, 0:2] < 0] = 0
            box[i][:, 2][box[i][:, 2] > w] = w
            box[i][:, 3][box[i][:, 3] > h] = h
            box_w = box[i][:, 2] - box[i][:, 0]
            box_h = box[i][:, 3] - box[i][:, 1]
            box[i] = box[i][np.logical_and(box_w > 1, box_h > 1)]
            box_data = np.zeros((len(box[i]), 5))
            box_data[:len(box[i])] = box[i]

        image_datas.append(image_data)
        box_datas.append(box_data)

    # 将图片分割，放在一起
    cutx = int(w * min_offset_x)
    cuty = int(h * min_offset_y)

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # 进行色域变换
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = cv2.cvtColor(np.array(new_image / 255, np.float32), cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0
    new_image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas, cutx, cuty)

    return new_image, new_boxes


'''worked'''
def get_random_data1(annotation_line, input_shape=None, random=True, hue=.1, sat=1.5, val=1.5, proc_img=True, image_root=None):
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
        box[i][:, [2]] = box[i][:, [0]] + box[i][:, [2]]
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
        cv2.imshow("draw" + str(i), image1)

        # 图片的大小
        iw, ih = image.size
        # 保存框的位置

        # image.save(str(index)+".jpg")
        # 是否翻转图片
        flip = rand() < .5
        if flip and len(box[i]) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[i][:, [0, 2]] = iw - box[i][:, [2, 0]]

        # 对输入进来的图片进行缩放
        new_ar = w / h
        scale = rand(scale_low, scale_high)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 进行色域变换
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

        image = Image.fromarray((image * 255).astype(np.uint8))
        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255
        # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")
        index = index + 1
        box_data = []
        # 对box进行重新处理

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
        # print(i)
        image_datas.append(image_data)
        box_datas.append(box_data)
        # print(box_datas)

        img = Image.fromarray((image_data * 255).astype(np.uint8))
        for j in range(len(box_data)):
            thickness = 3
            left, top, right, bottom = box_data[j][0:4]
            draw = ImageDraw.Draw(img)
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))

        # img.show()

    # 将图片分割，放在一起
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas, cutx, cuty)

    return new_image, new_boxes


'''Original'''
def get_random_data(annotation_line, input_shape, random=True, hue=.1, sat=1.5, val=1.5, proc_img=True, image_root=None):
    '''random preprocessing for real-time data augmentation'''
    h, w = input_shape
    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2

    image_datas = []
    box_datas = []
    index = 0

    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
    place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]

    with open(annotation_line, 'r') as f:
        lines = f.readlines()

        for line in lines:
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = Image.open(line_content[0])
            image = image.convert("RGB")
            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([np.array(list(map(int, map(float, (box.split(',')))))) for box in line_content[1:]])

            # image.save(str(index)+".jpg")
            # 是否翻转图片
            flip = rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # 对输入进来的图片进行缩放
            new_ar = w / h
            scale = rand(scale_low, scale_high)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 进行色域变换
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

            image = Image.fromarray((image * 255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image) / 255

            # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")

            index = index + 1
            box_data = []
            # 对box进行重新处理
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = abs(box[:, 2] - box[:, 0])
                box_h = abs(box[:, 3] - box[:, 1])
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

            img = Image.fromarray((image_data * 255).astype(np.uint8))
            for j in range(len(box_data)):
                thickness = 3
                left, top, right, bottom = box_data[j][0:4]
                draw = ImageDraw.Draw(img)
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
            img.show()

    # 将图片分割，放在一起
    cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas, cutx, cuty)

    return new_image, new_boxes


if __name__ == "__main__":
    ann_path = r"/datasets/cocosplit/seed1/full_box_10shot_skis_trainval.json"
    # ann_path = r"C:\Users\lenovo\Desktop\1.txt"
    image_root = r"J:\Beijing\FSCE\datasets\coco\val2014"

    mosaic_img, mosaic_ann = get_random_data1(ann_path, image_root=image_root)
    # mosaic_img, mosaic_ann = get_random_data_with_Mosaic(ann_path, image_root)
    # mosaic_img, mosaic_ann = get_random_data(ann_path, (480, 640), image_root=image_root)

    mosaic_img *= 255.
    mosaic_img = mosaic_img.astype(np.uint8)
    mosaic_img = cv2.cvtColor(mosaic_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("mosaic_img", mosaic_img)
    draw_1 = mosaic_img
    for i in range(len(mosaic_ann)):
        draw_1 = cv2.rectangle(draw_1, (int(mosaic_ann[i][0]), int(mosaic_ann[i][1])),
                               (int(mosaic_ann[i][2]), int(mosaic_ann[i][3])),
                               (0, 255, 255), 2)

    cv2.imshow("draw_1", draw_1)
    cv2.waitKey(0)

'''
Yolo-v5 内的实现  https://github.com/ultralytics/yolov5/blob/3ef3a95cfa536f3977676b6ed18f7bebf391fa2c/utils/datasets.py#L674
'''
# import cv2
# import math
# import torch
# import random
# import numpy as np
#
# def load_image(self, index):
#     # loads 1 image from dataset, returns img, original hw, resized hw
#     img = self.imgs[index]
#     if img is None:  # not cached
#         path = self.img_files[index]
#         img = cv2.imread(path)  # BGR
#         assert img is not None, 'Image Not Found ' + path
#         h0, w0 = img.shape[:2]  # orig hw
#         r = self.img_size / max(h0, w0)  # ratio
#         if r != 1:  # if sizes are not equal
#             img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
#                              interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
#         return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
#     else:
#         return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized
#
# def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
#     # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
#     w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
#     w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
#     ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
#     return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
#
# def resample_segments(segments, n=1000):
#     # Up-sample an (n,2) segment
#     for i, s in enumerate(segments):
#         x = np.linspace(0, len(s) - 1, n)
#         xp = np.arange(len(s))
#         segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
#     return segments
#
# def segment2box(segment, width=640, height=640):
#     # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
#     x, y = segment.T  # segment xy
#     inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
#     x, y, = x[inside], y[inside]
#     return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy
#
# def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
#     # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
#     y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
#     y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
#     y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
#     return y
#
# def xyn2xy(x, w=640, h=640, padw=0, padh=0):
#     # Convert normalized segments into pixel segments, shape (n,2)
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[:, 0] = w * x[:, 0] + padw  # top left x
#     y[:, 1] = h * x[:, 1] + padh  # top left y
#     return y
#
#
# def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
#                        border=(0, 0)):
#     # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
#     # targets = [cls, xyxy]
#
#     height = img.shape[0] + border[0] * 2  # shape(h,w,c)
#     width = img.shape[1] + border[1] * 2
#
#     # Center
#     C = np.eye(3)
#     C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
#     C[1, 2] = -img.shape[0] / 2  # y translation (pixels)
#
#     # Perspective
#     P = np.eye(3)
#     P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
#     P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
#
#     # Rotation and Scale
#     R = np.eye(3)
#     a = random.uniform(-degrees, degrees)
#     # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
#     s = random.uniform(1 - scale, 1 + scale)
#     # s = 2 ** random.uniform(-scale, scale)
#     R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
#
#     # Shear
#     S = np.eye(3)
#     S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
#     S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
#
#     # Translation
#     T = np.eye(3)
#     T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
#     T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)
#
#     # Combined rotation matrix
#     M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
#     if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
#         if perspective:
#             img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
#         else:  # affine
#             img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
#
#     # Visualize
#     # import matplotlib.pyplot as plt
#     # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
#     # ax[0].imshow(img[:, :, ::-1])  # base
#     # ax[1].imshow(img2[:, :, ::-1])  # warped
#
#     # Transform label coordinates
#     n = len(targets)
#     if n:
#         use_segments = any(x.any() for x in segments)
#         new = np.zeros((n, 4))
#         if use_segments:  # warp segments
#             segments = resample_segments(segments)  # upsample
#             for i, segment in enumerate(segments):
#                 xy = np.ones((len(segment), 3))
#                 xy[:, :2] = segment
#                 xy = xy @ M.T  # transform
#                 xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine
#
#                 # clip
#                 new[i] = segment2box(xy, width, height)
#
#         else:  # warp boxes
#             xy = np.ones((n * 4, 3))
#             xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
#             xy = xy @ M.T  # transform
#             xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine
#
#             # create new boxes
#             x = xy[:, [0, 2, 4, 6]]
#             y = xy[:, [1, 3, 5, 7]]
#             new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
#
#             # clip
#             new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
#             new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
#
#         # filter candidates
#         i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
#         targets = targets[i]
#         targets[:, 1:5] = new[i]
#
#     return img, targets
#
#
# def load_mosaic(self, index):
#     # loads images in a 4-mosaic
#
#     labels4, segments4 = [], []
#     s = self.img_size
#     yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
#     indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
#     for i, index in enumerate(indices):
#         # Load image
#         img, _, (h, w) = load_image(self, index)
#
#         # place img in img4
#         if i == 0:  # top left
#             img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
#             x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
#             x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
#         elif i == 1:  # top right
#             x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
#             x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
#         elif i == 2:  # bottom left
#             x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
#             x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
#         elif i == 3:  # bottom right
#             x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
#             x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
#
#         img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
#         padw = x1a - x1b
#         padh = y1a - y1b
#
#         # Labels
#         labels, segments = self.labels[index].copy(), self.segments[index].copy()
#         if labels.size:
#             labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
#             segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
#         labels4.append(labels)
#         segments4.extend(segments)
#
#     # Concat/clip labels
#     labels4 = np.concatenate(labels4, 0)
#     for x in (labels4[:, 1:], *segments4):
#         np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
#     # img4, labels4 = replicate(img4, labels4)  # replicate
#
#     # Augment
#     img4, labels4 = random_perspective(img4, labels4, segments4,
#                                        degrees=self.hyp['degrees'],
#                                        translate=self.hyp['translate'],
#                                        scale=self.hyp['scale'],
#                                        shear=self.hyp['shear'],
#                                        perspective=self.hyp['perspective'],
#                                        border=self.mosaic_border)  # border to remove
#
#     return img4, labels4
