#This program is used to remove the edge in the left side of the image and crop the blank before text
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

def energy_plot(image):
    H,W = image.shape
    gray_sum = np.sum(image,axis=0)
    plt.figure()
    plt.title("Pixel Histogram")
    plt.xlabel("X")
    plt.ylabel("sum of pixels")
    plt.plot(gray_sum)
    plt.xlim([0,W])
    plt.show()

def crop_blank(img):
    H,W = img.shape
    canny_img = ~cv2.Canny(img,100,200)
    energy_plot(canny_img)
    pixel_sum = np.sum(canny_img,axis=0)

    if len(pixel_sum) <= 2:
        return img
    idx = 0
    for x in range(len(pixel_sum)-2):
        isCrest = pixel_sum[x+1] >= pixel_sum[x] and pixel_sum[x+2] <= pixel_sum[x+1]
        isTrough = pixel_sum[x+1] <= pixel_sum[x] and pixel_sum[x+2] > pixel_sum[x+1]
        if isCrest and pixel_sum[x+1]/H < 235:
            return img
        if isTrough:
            idx = x+1
            break
    return img[:,idx::]

def canny_crop(edges,image):
    H, W = image.shape
    #THRESHOLD = 100 * W * 1.0 / H
    THRESHOLD = 4
    pixel_sum = np.sum(edges, axis=0)
    if len(pixel_sum) <= 2:
        return image

    idx = 0

    for x in range(1,len(pixel_sum) - 2):
        isCrest = pixel_sum[x+1] >= pixel_sum[x] and pixel_sum[x+2] <= pixel_sum[x+1]
        isTrough = pixel_sum[x+1] <= pixel_sum[x] and pixel_sum[x+2] >= pixel_sum[x+1]
        if isTrough:
            return image
        if isCrest and pixel_sum[x+1]/H > 225:
            idx = x + 1
            break

    if idx < THRESHOLD:
        return image[:, idx::]
    else:
        return image



if __name__=="__main__":
    img_path = './imgs/'
    save_path = './save_imgs/'
    img_path_list = glob.glob(img_path+'*.*')
    for img_path in img_path_list:
        gray_img = cv2.imread(img_path,0)
        edges = ~cv2.Canny(gray_img, 100, 200)
        crop_img = canny_crop(edges,gray_img)
        crop_img = crop_blank(crop_img)
        img_name = img_path.split('/')[-1]
        cv2.imwrite(save_path+img_name, crop_img)
        cv2.imshow('crop_img', crop_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
