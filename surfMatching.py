import cv2
import numpy as np
from time import time

def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d_SURF.create()
    #sift = cv2.xfeatures2d_SIFT.create()
    kp,des = surf.detectAndCompute(image,None)
    #kp,des = sift.detectAndCompute(image,None)
    kp_image = cv2.drawKeypoints(gray_image,kp,None)
    return kp_image,kp,des


def get_good_match(des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def siftImageAlignment(img1,img2):
   _,kp1,des1 = sift_kp(img1)
   _,kp2,des2 = sift_kp(img2)
   goodMatch = get_good_match(des1,des2)
   if len(goodMatch) > 4:
       ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ransacReprojThreshold = 4
       H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
       imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
   return imgOut,H,status



img1 = cv2.imread('./img/2.jpg')
img2 = cv2.imread('./img/3.jpg')
print(img1.shape)
print(img2.shape)


start = time()
result,_,_ = siftImageAlignment(img1,img2)

print(result.shape)
end = time()
print('time cost is: ',end - start)
cv2.imshow('result',result)
cv2.waitKey(0)

cv2.imshow('img1',img1)
cv2.waitKey(0)

cv2.imshow('img2',img2)
cv2.waitKey(0)
