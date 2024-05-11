import numpy as np
from numpy import linalg as la
from scipy.stats import entropy
import matplotlib.pyplot as plt
import cv2
import pywt
import os

if __name__ == '__main__':
    M = 512  # 固定原圖像長寬
    N = 128  # 固定浮水印長寬
    K = 64  # 切割區塊大小

    alpha = 0.1  # 強度係數

    # 原始圖檔的放置位置
    path = './origin_image'
    if not os.path.isdir(path):
        os.makedirs(path)

    origin_pictures = []
    watered_pictures = []
    for root, dirs, files in os.walk("origin_image"):
        for file in files:
            origin_pictures.append(file)

    for i in range(len(origin_pictures)):
        # 匯入圖檔、浮水印圖
        I = cv2.imread("./origin_image/" + origin_pictures[i])  # 希望加入的圖(origin picture)
        G = cv2.imread("7.1.03.tiff")  # 浮水印(watermark)
        # W = np.zeros(M)

        # 縮放、轉灰階圖、(改變矩陣精度)
        I = cv2.resize(I, (M, M))
        # I = Image.fromarray(I).resize(size=(M, M))
        # I = cv2.im2double(I); # double精度轉換
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)  # 灰階化處理

        # G = Image.fromarray(G).resize(size=(N, N))
        G = cv2.resize(G, (N, N))
        # G = cv2.im2double(G) # double精度轉換
        G = cv2.cvtColor(G, cv2.COLOR_BGR2RGB)
        G = cv2.cvtColor(G, cv2.COLOR_RGB2GRAY)  # 灰階化處理

        # 顯示置入
        plt.subplot(2, 2, 1)
        plt.imshow(I, cmap="gray")
        plt.title("image")
        plt.subplot(2, 2, 2)
        plt.imshow(G, cmap="gray")
        plt.title("watermark")

        # Step 1
        LL, (LH, HL, HH) = pywt.dwt2(G, 'haar')  # 做 2 維的 haar DWT 轉換
        [U, S, V] = la.svd(HH)  # 對 HH 做 SVD 分解，取 U、S、V 矩陣

        # Step 2
        # 做 2 級 haar DWT 轉換
        LL1, (LH1, HL1, HH1) = pywt.dwt2(I, 'haar')
        LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'haar')  # 128*128
        # H0 = entropy(HH2)  # 計算HH3系数的熵
        # Step 3

        # 選擇最佳嵌入區塊 默認是4 * 4: (1, 1)
        optimal_block_index = 0

        # Step 4
        # 對選擇的嵌入區塊進行 DCT 轉換, 得到DCT係數矩陣 B
        m = np.floor(optimal_block_index / 4) + 1
        n = np.mod(optimal_block_index, 4) + 1
        x = (m - 1) * K + 1
        y = (n - 1) * K + 1
        H_I = HH2[int(x): int(x + K), int(y): int(y + K)]
        B = cv2.dct(np.float32(H_I))

        # Step 5
        # 對 B 做奇異值分解，再嵌入浮水印
        U1, S1, V1 = la.svd(B)
        S2 = S1 + alpha * S
        B1 = U1 * S2 * V1
        H_I = cv2.idct(B1)
        HH2[int(x): int(x + K), int(y):int(y + K)] = H_I
        LL1 = pywt.idwt2((LL2, (LH2, HL2, HH2)), 'haar')
        W = pywt.idwt2((LL1, (LH1, HL1, HH1)), 'haar')
        W = np.uint8(W)

        plt.subplot(2, 2, 3)
        plt.imshow(W, cmap="gray")
        plt.title('img_watermarked')

        plt.show()

        watered_pictures.append(W)

    # save all picture change to the file
    watered_path = './watered_image/'
    if not os.path.isdir(watered_path):
        os.makedirs(watered_path)

    for i in range(len(watered_pictures)):
        print(origin_pictures[i])
        cv2.imwrite(watered_path + "watered_" + origin_pictures[i], watered_pictures[i])
