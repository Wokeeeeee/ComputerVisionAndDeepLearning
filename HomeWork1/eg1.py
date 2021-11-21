import numpy as np
import cv2
import matplotlib.pyplot as plt

grey_img = cv2.imread("lena512.bmp", cv2.IMREAD_GRAYSCALE)
color_img = cv2.imread("lena512color.tiff", cv2.IMREAD_COLOR)


for window_size in range(3,31,2):
    for sigma in range(1, 13, 2):
        print("window size=", window_size,"   segma=", sigma)
        # 高斯滤波降噪
        grey_blur = cv2.GaussianBlur(grey_img, (window_size, window_size), 0)
        color_blur = cv2.GaussianBlur(color_img, (window_size, window_size), 0)
        # 高斯拉普拉斯检测子
        grey_laplacian = cv2.Laplacian(grey_blur, -1, cv2.CV_64F, ksize=sigma)
        color_laplacian = cv2.Laplacian(color_blur, -1, cv2.CV_64F, ksize=sigma)
        plt.subplot(3, 2, (int)(sigma + 1) / 2)
        plt.imshow(grey_laplacian)
        plt.xticks([])
        plt.yticks([])
    plt.show()