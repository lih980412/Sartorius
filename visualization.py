import pandas as pd
import numpy as np
import cv2


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


def build_masks(df_train, image_ids, input_shape):
    height, width = input_shape
    labels = df_train[df_train["id"] == image_ids[0]]["annotation"].tolist()
    mask = np.zeros((height, width))
    for label in labels:
        mask += rle_decode(label, shape=(height, width))
    mask = mask.clip(0, 1)  # 大于0被替换成0，小于1被替换成1
    return mask


if __name__ == "__main__":
    data = pd.read_excel(r"K:\\LiHang\\Cell Instance Segmentation\\train_3.csv")
    img = cv2.imread(r"K:\\LiHang\\Cell Instance Segmentation\\train\\0030fd0e6378.png")

    cv2.imshow("img", img)

    image_ids = data.id.unique().tolist()
    mask = build_masks(data, image_ids, (520, 704))
    cv2.imshow("mask", mask)
    mask = (mask*255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (128, 0, 128), 1)
    # img = img[:, :, ::-1]   # BGR 转成 RGB
    img[..., 0] = np.where(mask == 255, 150, img[..., 0])
    cv2.imshow("mask_Img", img)
    cv2.waitKey(0)

    # fig = plt.figure()
    # ax1 = plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # ax2 = plt.subplot(1, 2, 2)
    # plt.imshow(mask)
    # # plt.show()
    #
    # pylab.show()
