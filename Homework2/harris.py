import cv2
import numpy as np


def opencv_harris():
    img = cv2.imread('./check_board.png')
    img = cv2.resize(img, dsize=(600, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 3, 3, 0.04)
    print(dst.shape)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('cv', img)


def harris_detect(img, ksize=3):
    k = 0.04
    threshold = 0.01
    WITH_NMS = False

    #使用Sobel计算像素点x,y方向的梯度
    h, w = img.shape[:2]
    grad = np.zeros((h, w, 2), dtype=np.float32)
    Ix = np.zeros((h, w), dtype=np.float32)
    Iy = np.zeros((h, w), dtype=np.float32)
    Ix = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    # 求得Ix^2 Iy^2 IxIy并高斯滤波
    I2x = cv2.GaussianBlur(Ix ** 2, ksize=(ksize, ksize), sigmaX=2)
    I2y = cv2.GaussianBlur(Iy ** 2, ksize=(ksize, ksize), sigmaX=2)
    IxIy = cv2.GaussianBlur(Ix * Iy, ksize=(ksize, ksize), sigmaX=2)

    #    Ix^2 IxIy
    # M= IxIy Iy^2 求和
    M = [np.array([[I2x[i, j], IxIy[i, j]], [IxIy[i, j], I2y[i, j]]]) for i in range(h) for j in range(w)]
    det_ = list(map(np.linalg.det, M))
    trace_ = list(map(np.trace, M))
    # R(i,j)=det(M)-k(trace(M))^2
    R = np.array([d - k * t ** 2 for d, t in zip(det_, trace_)] )
    # R = np.array([d - k * t ** 2 for d, t in zip(det_, trace_)])
    R_max = np.max(R)
    R = R.reshape(h, w)
    corner = np.zeros_like(R, dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if R[i, j] > R_max * threshold:
                corner[i, j] = 255
    return corner


if __name__ == '__main__':
    img = cv2.imread("./check_board.png")
    img = cv2.resize(img, dsize=(600, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # my harris
    dst = harris_detect(gray)
    print(dst.shape)  # (400, 600)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('my', img)
    #opencv
    opencv_harris()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
