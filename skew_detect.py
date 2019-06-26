
import numpy as np
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.transform import hough_line, hough_line_peaks, rotate


class SkewDetect:

    piby4 = np.pi / 4

    def __init__(
        self,
        sigma=3.0,
        num_peaks=20,
    ):

        self.sigma = sigma
        self.num_peaks = num_peaks

    def get_max_freq_elem(self, arr):

        max_arr = []
        freqs = {}
        for i in arr:
            if i in freqs:
                freqs[i] += 1
            else:
                freqs[i] = 1

        sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
        max_freq = freqs[sorted_keys[0]]

        key_arr = list()
        for k in sorted_keys:
            if freqs[k] == max_freq:
                key_arr.append(k)
        key_min = np.min(key_arr)
        max_arr.append(key_min)
        for k in sorted_keys:
            if k != key_min and freqs[k] == max_freq and k - key_min < 45:
                max_arr.append(k)

        return max_arr

    def compare_sum(self, value):
        if value >= 44 and value <= 46:
            return True
        else:
            return False

    def calculate_deviation(self, angle):

        angle_in_degrees = np.abs(angle)
        deviation = np.abs(SkewDetect.piby4 - angle_in_degrees)

        return deviation

    def determine_skew(self, img):

        img_gray = rgb2gray(img)
        edges = canny(img_gray, sigma=self.sigma)
        h, a, d = hough_line(edges)
        _, ap, _ = hough_line_peaks(h, a, d, num_peaks=self.num_peaks)

        if len(ap) == 0:
            raise Exception

        ap_deg = np.rad2deg(ap)
        ap_deg = np.round(- ap_deg + 90)

        angles = list()
        angle0045 = (ap_deg[ap_deg < 45])[ap_deg[ap_deg < 45] >= 0]
        angle4590 = (ap_deg[ap_deg < 90])[ap_deg[ap_deg < 90] >= 45]
        angle90135 = (ap_deg[ap_deg < 135])[ap_deg[ap_deg < 135] >= 90]
        angle135180 = (ap_deg[ap_deg < 180])[ap_deg[ap_deg < 180] >= 135]
        angles.append(angle0045)
        angles.append(angle4590)
        angles.append(angle90135)
        angles.append(angle135180)
        region_len = np.array([len(angle0045), len(angle4590), len(angle90135), len(angle135180)])
        region = np.argmax(region_len)

        ans_arr = self.get_max_freq_elem(angles[region])
        ans_res = np.mean(ans_arr)

        img = rotate(img, - ans_res, resize=True)

        return img

def get_deskewed(img):
    deskew = SkewDetect()
    img = deskew.determine_skew(img)

    return img

