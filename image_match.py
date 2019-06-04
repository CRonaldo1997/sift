import numpy as np
import cv2
from time import time
from matplotlib import pyplot as plt


class MatchImage:

    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.MIN_MATCH_COUNT = 4
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.draw_switch = False

    def get_bbox(self, quary_img, target_img):
        kp1, des1 = self.sift.detectAndCompute(quary_img, None)
        kp2, des2 = self.sift.detectAndCompute(target_img, None)
        matches = self.flann.knnMatch(des1, des2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.65 * n.distance:
                good.append(m)

        if len(good) >= self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4)
            if M is not None:
                h, w = quary_img.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                if self.draw_switch:
                    matchesMask = mask.ravel().tolist()
                    draw_params = dict(matchColor=(0, 255, 0),
                                       singlePointColor=None,
                                       matchesMask=matchesMask,
                                       flags=2)
                    img3 = cv2.drawMatches(quary_img, kp1, target_img, kp2, good, None, **draw_params)
                    plt.imshow(img3, 'gray'), plt.show()

                return tuple(dst[0][0]), tuple(dst[2][0])


if __name__ == '__main__':
    matchImage = MatchImage()
    quary_img = cv2.imread('6.png', 0)
    target_img = cv2.imread('5.png', 0)
    start = time()
    result = matchImage.get_bbox(quary_img, target_img)
    print('time cost: ', time() - start)

    if result:
        left_top, right_bottom = result
        target_img = cv2.rectangle(target_img, left_top, right_bottom, color=(0, 255, 0), thickness=4)
        cv2.imshow('bbox', target_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('match failed!')
