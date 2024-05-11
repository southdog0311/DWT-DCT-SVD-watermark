import math
import numpy as np
from numpy import linalg as la
from scipy.stats import entropy
import matplotlib.pyplot as plt
import cv2
import pywt
import os
from skimage.metrics import structural_similarity as ssim

# 攻击

def add_gaussian_noise(image, mean=0, sigma=25):
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image

# W = add_gaussian_noise(W)

#salt pepper
def add_salt_pepper_noise(image, prob=0.05):
    noisy_image = np.copy(image)
    # 添加椒鹽噪聲
    num_salt = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    num_pepper = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image

# W = add_salt_pepper_noise(W)

# 定義一個高通濾波器函數
def high_pass_filter(image, kernel_size=3):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

# W = high_pass_filter(W)

# 定義一個均值濾波器函數
def mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))

# W = mean_filter(W)

# Define a median filter function
def median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

# W = median_filter(W)

def rotate_image(image, angle):
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    return rotated_image

# Rotate the images
# angle = 20  # Define the rotation angle
# W = rotate_image(W, angle)

# Crop the images
def crop_image(image, crop_ratio_height, crop_ratio_width):
    h, w = image.shape[:2]
    h_crop = int(h * crop_ratio_height)
    w_crop = int(w * crop_ratio_width)
    x_start = np.random.randint(0, h - h_crop)
    y_start = np.random.randint(0, w - w_crop)
    cropped_image = np.copy(image)
    cropped_image[x_start:x_start + h_crop, y_start:y_start + w_crop] = 0
    return cropped_image

# crop_ratio_height = 0.3  # Define the crop ratio for height
# crop_ratio_width = 0.3  # Define the crop ratio for width
# W = crop_image(W, crop_ratio_height, crop_ratio_width)

def compress_image(image, quality):
    temp_path = 'temp.jpg'
    cv2.imwrite(temp_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    compressed_image = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
    return compressed_image

# quality = 50  # Define the compression quality factor
# W = compress_image(W, quality)

origin_pictures = []
watered_pictures = []
recover_pictures = []
for root, dirs, files in os.walk("origin_image"):
    for file in files:
        origin_pictures.append(file)
for root, dirs, files in os.walk("watered_image"):
    for file in files:
        watered_pictures.append(file)
for root, dirs, files in os.walk("extract_image"):
    for file in files:
        recover_pictures.append(file)

M = 512
N = 128
nc_value_sum = 0
psnr_val_sum = 0
ssim_value_sum = 0
bert_value_sum = 0
mse_value_sum = 0
for i in range(len(watered_pictures)):
    I = cv2.imread("./origin_image/" + origin_pictures[i])
    I = cv2.resize(I, (M, M))
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
    G = cv2.imread("7.1.03.tiff")  # 浮水印(watermark)
    G = cv2.resize(G, (N, N))
    G = cv2.cvtColor(G, cv2.COLOR_BGR2RGB)
    G = cv2.cvtColor(G, cv2.COLOR_RGB2GRAY)
    W = cv2.imread("./watered_image/" + watered_pictures[i])
    W = cv2.resize(W, (M, M))
    W = cv2.cvtColor(W, cv2.COLOR_BGR2RGB)
    W = cv2.cvtColor(W, cv2.COLOR_RGB2GRAY)
    A = cv2.imread("./extract_image/" + recover_pictures[i], 0)

    Attack = list()
    Attack.append(add_gaussian_noise(W.copy()))
    Attack.append(add_salt_pepper_noise(W.copy()))
    Attack.append(high_pass_filter(W.copy()))
    Attack.append(mean_filter(W.copy()))
    Attack.append(median_filter(W.copy()))
    # Rotate the images
    angle = 20  # Define the rotation angle
    Attack.append(rotate_image(W.copy(), angle))
    crop_ratio_height = 0.3  # Define the crop ratio for height
    crop_ratio_width = 0.3  # Define the crop ratio for width
    Attack.append(crop_image(W.copy(), crop_ratio_height, crop_ratio_width))
    quality = 50  # Define the compression quality factor
    Attack.append(compress_image(W.copy(), quality))

    for j in range(1):
        print("attack", j+1)
        K = 64  # 切割區塊大小
        alpha = 0.1  # 強度係數
        # 選擇最佳嵌入區塊 默認是4 * 4: (1, 1)
        optimal_block_index = 0

        LL, (LH, HL, HH) = pywt.dwt2(G, 'haar')  # 做 2 維的 haar DWT 轉換
        [U, S, V] = la.svd(HH)  # 對 HH 做 SVD 分解，取 U、S、V 矩陣

        # 做 2 級 haar DWT 轉換
        LL1, (LH1, HL1, HH1) = pywt.dwt2(I, 'haar')
        LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'haar')  # 128*128

        # 對選擇的嵌入區塊進行 DCT 轉換, 得到DCT係數矩陣 B
        m = np.floor(optimal_block_index / 4) + 1
        n = np.mod(optimal_block_index, 4) + 1
        x = (m - 1) * K + 1
        y = (n - 1) * K + 1
        H_I = HH2[int(x): int(x + K), int(y): int(y + K)]
        B = cv2.dct(np.float32(H_I))

        U1, S1, V1 = la.svd(B)

        # 提取水印
        LL3, (LH3, HL3, HH3) = pywt.dwt2(Attack[0], 'haar')
        LL4, (LH4, HL4, HH4) = pywt.dwt2(LL3, 'haar')  # 128*128
        H_I2 = HH4[int(x):int(x + K), int(y):int(y + K)]
        B2 = cv2.dct(np.float32(H_I2))
        Uw, Sw, Vw = la.svd(B2)
        Sx = (Sw - S1) / alpha
        B2 = U * Sx * V
        H_I2 = cv2.idct(B2)
        A = pywt.idwt2((LL, (LH, HL, H_I2)), 'haar')
        A = np.uint8(A)

        plt.subplot(2, 2, 1)
        plt.imshow(G, cmap="gray")
        plt.title('origin')

        plt.subplot(2, 2, 2)
        plt.imshow(A, cmap="gray")
        plt.title('recover')

        plt.show()

        def calculate_histograms_and_nc(image1, image2):
            h1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
            h2 = cv2.calcHist([image2], [0], None, [256], [0, 256])

            # Plot the histograms
            # plt.plot(h1, color='b', label='Image 1')
            # plt.plot(h2, color='r', label='Image 2')
            # plt.xlabel('Intensity')
            # plt.ylabel('Frequency')
            # plt.title('Histogram Comparison')
            # plt.legend()
            # plt.show()

            # Calculate NC value based on histograms
            numerator = np.sum(np.sqrt(h1 * h2))
            denominator = np.sqrt(np.sum(h1) * np.sum(h2))
            nc_val = numerator / denominator

            # Return the NC value
            return nc_val


        nc_value = calculate_histograms_and_nc(G, A)
        nc_value_sum = nc_value_sum + nc_value


        def psnr(img1, img2):
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return "Identical images"
            PIXEL_MAX = 255.0
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


        # 計算 PSNR
        psnr_val = psnr(G, A)
        psnr_val_sum = psnr_val_sum + psnr_val
        # 顯示 PSNR 值
        print('The PSNR value is: ', psnr_val)


        def calculate_ssim(image1, image2, win_size=7):
            ssim_value = ssim(image1, image2, win_size=win_size)
            return ssim_value


        ssim_value = calculate_ssim(G, A, win_size=7)
        ssim_value_sum = ssim_value_sum + ssim_value
        print('The SSIM value between the two images is', ssim_value, '.')


        def calculate_bert(img1, img2):
            if img1.shape != img2.shape:
                raise ValueError("Images are not of the same shape.")

            error_count = np.sum(img1 != img2)
            total_pixels = img1.shape[0] * img1.shape[1]

            bert = error_count / total_pixels
            return bert


        bert_value = calculate_bert(G, A)
        bert_value_sum = bert_value_sum + bert_value
        print(f"BERT值為: {bert_value * 100}%")


        def calculate_mse(image1, image2):
            if image1.shape != image2.shape:
                raise ValueError("Images are not of the same shape.")
            mse = np.mean((image1 - image2) ** 2)
            return mse


        mse_value = calculate_mse(G, A)
        mse_value_sum = mse_value_sum + mse_value
        print(f"MSE 值為: {mse_value}\n")

    print("No.", i)
    print("total: ")
    print('The NC value is:', nc_value_sum / (len(watered_pictures)))
    print('The PSNR value is: ', psnr_val_sum / (len(watered_pictures)))
    print('The SSIM value is', ssim_value_sum / (len(watered_pictures)))
    print(f"BERT value is: {bert_value_sum * 100 / (len(watered_pictures))}%")
    print(f"MSE value is: {mse_value_sum / (len(watered_pictures))}")


# quality = 50  # Define the compression quality factor
# W = compress_image(W, quality)

# N = 128
# watered_pictures = []
# recover_pictures = []
# for root, dirs, files in os.walk("extract_image"):
#     for file in files:
#         recover_pictures.append(file)
# for root, dirs, files in os.walk("watered_image"):
#     for file in files:
#         watered_pictures.append(file)
#
# G = cv2.imread("7.1.03.tiff")  # 浮水印(watermark)
# G = cv2.resize(G, (N, N))
# G = cv2.cvtColor(G, cv2.COLOR_BGR2RGB)
# G = cv2.cvtColor(G, cv2.COLOR_RGB2GRAY)
# W = cv2.imread("./watered_image/" + watered_pictures[0])
# A = cv2.imread("./extract_image/" + recover_pictures[0], 0)
#
#
# def calculate_histograms_and_nc(image1, image2):
#     h1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
#     h2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
#
#     # Plot the histograms
#     plt.plot(h1, color='b', label='Image 1')
#     plt.plot(h2, color='r', label='Image 2')
#     plt.xlabel('Intensity')
#     plt.ylabel('Frequency')
#     plt.title('Histogram Comparison')
#     plt.legend()
#     plt.show()
#
#     # Calculate NC value based on histograms
#     numerator = np.sum(np.sqrt(h1 * h2))
#     denominator = np.sqrt(np.sum(h1) * np.sum(h2))
#     nc_val = numerator / denominator
#
#     # Return the NC value
#     return nc_val
#
#
# # 使用函數
# nc_value = calculate_histograms_and_nc(G, A)
# print('The NC value between the two images is:', nc_value)
#
#
# def psnr(img1, img2):
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return "Identical images"
#     PIXEL_MAX = 255.0
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
#
#
# # 計算 PSNR
# psnr_val = psnr(G, A)
#
# # 顯示 PSNR 值
# print('The PSNR value is: ', psnr_val)
#
#
# def calculate_ssim(image1, image2, win_size=7):
#     ssim_value = ssim(image1, image2, win_size=win_size)
#     return ssim_value
#
#
# # 使用函數
# ssim_value = calculate_ssim(G, A, win_size=7)
# print('The SSIM value between the two images is', ssim_value, '.')
#
#
# def calculate_bert(img1, img2):
#     if img1.shape != img2.shape:
#         raise ValueError("Images are not of the same shape.")
#
#     error_count = np.sum(img1 != img2)
#     total_pixels = img1.shape[0] * img1.shape[1]
#
#     bert = error_count / total_pixels
#     return bert
#
#
# bert_value = calculate_bert(G, A)
# print(f"BERT值為: {bert_value * 100}%")
#
#
# def calculate_mse(image1, image2):
#     if image1.shape != image2.shape:
#         raise ValueError("Images are not of the same shape.")
#
#     mse = np.mean((image1 - image2) ** 2)
#     return mse
#
#
# # 舉例
# mse_value = calculate_mse(G, A)
# print(f"MSE 值為: {mse_value}")
