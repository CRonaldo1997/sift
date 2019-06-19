from __future__ import division
import os
from PIL import Image
import xml.dom.minidom
import numpy as np
import traceback
from glob import glob
import cv2
import pathlib
import imgaug as ia
from imgaug import augmenters as iaa
import random
import math
from uuid import uuid1

iter = 8

seq = iaa.Sequential(
    [
        iaa.Affine(
            scale={"x": (0.99, 1.01), "y": (0.99, 1.01)},
            # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.02, 0.02), "y": (-0.01, 0.01)},
            # translate by -20 to +20 percent (per axis)
            rotate=(-2, 2),  # rotate by -45 to +45 degrees
            # shear=(-16, 16), # shear by -16 to +16 degrees
            #order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            order=[1],  # use bilinear interpolation (fast)
            # order=[1],
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ),
        # iaa.Add((-10, 10), per_channel=0.5),
        # iaa.AddToHueAndSaturation((-20, 20)),
        # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
        # iaa.FrequencyNoiseAlpha(
        #     exponent=(-4, 0),
        #     first=iaa.Multiply((0.5, 1.5), per_channel=True),
        #     second=iaa.ContrastNormalization((0.5, 2.0))
        # ),
        # iaa.ElasticTransformation(alpha=(0.25, 4), sigma=0.25),
        # iaa.ElasticTransformation(alpha=(0.25, 4), sigma=0.25),
        iaa.PiecewiseAffine(scale=(0.002, 0.005)),
    ],
    random_order=True
)


def adjust_gamma(image):
    gamma = random.randrange(8, 18, 2) / 10.0
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def resize2standard(img_arr, WS = 256, HS = 32):
    ave = np.mean(img_arr)+10
    H,W,C = img_arr.shape
    new_W = math.ceil(1.0*H*WS/HS)
    if new_W > W:
        blank_img = ave*np.ones((H, new_W, C))
        blank_img[:,:W] = img_arr
        blank_img = cv2.resize(blank_img,(WS,HS))
        # cv2.imwrite('./pad.jpg',blank_img)
        return blank_img
    else:
        return cv2.resize(img_arr,(WS,HS))


arg_switch = True

ImgPath = './labeled/'
AnnoPath = './labeled/'
ProcessedPath = './img_piece/'

if not os.path.exists(ProcessedPath):
    os.makedirs(ProcessedPath)

imagelist = os.listdir(ImgPath)
imagelist = glob(ImgPath+'*.jpg')

for index in range(iter):
    for image in imagelist:
        print('a new image:', image)
        image_pre, ext = os.path.splitext(image)
        imgfile = image
        img_name = pathlib.PosixPath(imgfile).name
        xmlfile = image.replace('jpg','xml')

        DomTree = xml.dom.minidom.parse(xmlfile)
        annotation = DomTree.documentElement

        filenamelist = annotation.getElementsByTagName('filename')  # [<DOM Element: filename at 0x381f788>]
        filename = filenamelist[0].childNodes[0].data
        objectlist = annotation.getElementsByTagName('object')

        i = 1
        for idx, objects in enumerate(objectlist):

            namelist = objects.getElementsByTagName('name')
            objectname = namelist[0].childNodes[0].data
            print(objectname)
            bndbox = objects.getElementsByTagName('bndbox')
            for box in bndbox:
                try:
                    x1_list = box.getElementsByTagName('xmin')
                    x1 = int(x1_list[0].childNodes[0].data)
                    y1_list = box.getElementsByTagName('ymin')
                    y1 = int(y1_list[0].childNodes[0].data)
                    x2_list = box.getElementsByTagName('xmax')
                    x2 = int(x2_list[0].childNodes[0].data)
                    y2_list = box.getElementsByTagName('ymax')
                    y2 = int(y2_list[0].childNodes[0].data)
                    w = x2 - x1
                    h = y2 - y1
                    img_arr = cv2.imread(imgfile)
                    img_roi = img_arr[y1:y2, x1:x2]

                    # cv2.imshow(objectname,img_roi)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    if arg_switch:
                        img_roi = seq.augment_images(np.expand_dims(img_roi,axis=0))[0]
                    img_roi = adjust_gamma(img_roi)
                    # cv2.imshow(objectname,img_roi)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # dst_path = ProcessedPath+str(index)+str(idx)+img_name
                    dst_path = ProcessedPath + str(uuid1())+ img_name
                    img_roi = resize2standard(img_roi)
                    cv2.imwrite(dst_path, img_roi)
                    with open(dst_path.replace('jpg', 'txt'), 'w') as f:
                        f.write(objectname)
                except Exception as e:
                    traceback.print_exc()
