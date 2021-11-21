import math
import cv2
import numpy as np

def NMS(magnitude:np.ndarray,orientation:np.ndarray)->np.ndarray:
    result = np.zeros(magnitude.shape)  # 非极大值抑制图像矩阵
    for i in range(1, result.shape[0] - 1):
        for j in range(1, result.shape[1] - 1):
            if (orientation[i, j] == 0.0) and (magnitude[i, j] == np.max([magnitude[i, j], magnitude[i + 1, j], magnitude[i - 1, j]])):
                result[i, j] = magnitude[i, j]
            if (orientation[i, j] == -45.0) and magnitude[i, j] == np.max([magnitude[i, j], magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]):
                result[i, j] = magnitude[i, j]
            if (orientation[i, j] == 90.0) and magnitude[i, j] == np.max([magnitude[i, j], magnitude[i, j + 1], magnitude[i, j - 1]]):
                result[i, j] = magnitude[i, j]
            if (orientation[i, j] == 45.0) and magnitude[i, j] == np.max([magnitude[i, j], magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]):
                result[i, j] = magnitude[i, j]
    return result



def calculateGradient(image:np.ndarray):
    width, height = image.shape
    dx = cv2.Sobel(image, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_REPLICATE)
    dy = cv2.Sobel(image, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_REPLICATE)
    magnitude = np.zeros([width - 1, height - 1])#梯度幅度
    orientation = np.zeros([width - 1, height - 1])#梯度方向
    magnitude=np.sqrt(cv2.add(cv2.multiply(dx,dx),cv2.multiply(dy,dy)))
    orientation=cv2.divide(dx,dy)


    # 进行梯度量化
    for i in range(1, width - 2):
        for j in range(1, height - 2):
            if (((orientation[i, j] >= -22.5) and (orientation[i, j] < 22.5)) or
                    ((orientation[i, j] <= -157.5) and (orientation[i, j] >= -180)) or
                    ((orientation[i, j] >= 157.5) and (orientation[i, j] < 180))):
                orientation[i, j] = 0.0
            elif (((orientation[i, j] >= 22.5) and (orientation[i, j] < 67.5)) or
                  ((orientation[i, j] <= -112.5) and (orientation[i, j] >= -157.5))):
                orientation[i, j] = -45.0
            elif (((orientation[i, j] >= 67.5) and (orientation[i, j] < 112.5)) or
                  ((orientation[i, j] <= -67.5) and (orientation[i, j] >= -112.5))):
                orientation[i, j] = 90.0
            elif (((orientation[i, j] >= 112.5) and (orientation[i, j] < 157.5)) or
                  ((orientation[i, j] <= -22.5) and (orientation[i, j] >= -67.5))):
                orientation[i, j] = 45.0

    return magnitude, orientation

def double_threshold(image: np.ndarray, low:int, high:int) -> np.ndarray:
    image_threshold = np.zeros(image.shape)
    for i in range(1, image_threshold.shape[0] - 1):
        for j in range(1, image_threshold.shape[1] - 1):
            if image[i, j] < low:
                image_threshold[i, j] = 0
            elif image[i, j] > high:
                image_threshold[i, j] = 255
            elif ((image[i + 1, j] < high) or (image[i - 1, j] < high) or (image[i, j + 1] < high) or
                  (image[i, j - 1] < high) or (image[i - 1, j - 1] < high) or (image[i - 1, j + 1] < high) or
                  (image[i + 1, j + 1] < high) or (image[i + 1, j - 1] < high)):
                image_threshold[i, j] = 255
    return image_threshold


def canny(image: np.ndarray, threshold_low: int, threshold_high: int)->np.ndarray:
    # 1.使用高斯滤波降噪
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # 2.确定梯度幅值和方向
    magnitude,orientation=calculateGradient(image)
    # 3.非极大值抑制
    image_nms = NMS(magnitude, orientation)
    # 4.双阈值检测
    image_threshold = double_threshold(image_nms, threshold_low, threshold_high)
    return image_threshold


if __name__ == "__main__":
    img = cv2.imread("lena512.bmp", cv2.IMREAD_GRAYSCALE)
    my_canny_edge = canny(img, 50, 200)
    standard_canny_edge = cv2.Canny(img, 50, 200)
    cv2.imshow("my_canny_edge", my_canny_edge)
    cv2.imshow("standard_canny_edge", standard_canny_edge)
    cv2.waitKey(0)


