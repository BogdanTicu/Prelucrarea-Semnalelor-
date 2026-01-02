import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import ascent
from scipy.fft import dctn, idctn
from skimage import data
import imageio.v3 as iio
X = ascent()
X1 = data.astronaut().astype(np.float32) #pt ex2,3 am luat o imagine RGB.
# Matricea de cuantizare JPEG
Q_jpeg = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 28, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
])

def MSE(X, X_rec):
    return np.mean((X - X_rec) ** 2)
def JPEG(X, alpha=1):
    H, W = X.shape
    Q_scalat = Q_jpeg * alpha
    X_jpeg = np.zeros(X.shape)
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            block = X[i:i+8, j:j+8]
            y = dctn(block, norm = 'ortho')
            y_jpeg = Q_scalat*np.round(y / Q_scalat)
            x_jpeg = idctn(y_jpeg, norm = 'ortho')
            X_jpeg[i:i+8, j:j+8] = x_jpeg
    return X_jpeg


def rgb2ycbcr(X):
    R,G,B = X[:,:,0], X[:,:,1], X[:,:,2]
    Y  =  0.299*R + 0.587*G +0.114*B
    Cb = -0.168736*R - 0.331264*G + 0.5*B + 128
    Cr =  0.5*R - 0.418688*G - 0.081312*B + 128
    Ycbcr_img = np.zeros(X.shape)
    Ycbcr_img[:,:,0] = Y
    Ycbcr_img[:,:,1] = Cb
    Ycbcr_img[:,:,2] = Cr
    return Ycbcr_img

def ycbcr2rgb(X):
    Y,Cb,Cr = X[:,:,0], X[:,:,1], X[:,:,2]
    R = Y + 1.402*(Cr-128)
    G = Y - 0.344136*(Cb-128) - 0.714136*(Cr-128)
    B = Y + 1.772*(Cb-128)
    rgb_img = np.zeros(X.shape)
    rgb_img[:,:,0] = R
    rgb_img[:,:,1] = G
    rgb_img[:,:,2] = B
    return rgb_img

def compress_to_MSE(X, target_MSE):
    left = 1
    right = 100
    best_alpha = right
    while left <= right:
        mid = (left + right) // 2
        X_jpeg = JPEG(X, alpha=mid)
        mse = MSE(X, X_jpeg)
        if mse > target_MSE:

            right = mid - 1
        else:
            left = mid + 1
            best_alpha = mid
    X_jpeg = JPEG(X, alpha=best_alpha)
    return X_jpeg, best_alpha

def ex1():
    X_jpeg = JPEG(X)
    plt.subplot(121).imshow(X, cmap='gray', vmin=X.min(), vmax=X.max())
    plt.subplot(122).imshow(X_jpeg, cmap='gray', vmin=X_jpeg.min(), vmax=X_jpeg.max())
    plt.axis("off")
    plt.show()

def ex2(X):
    y = rgb2ycbcr(X)
    Y_jpeg = JPEG(y[:,:,0])
    Cb_jpeg = JPEG(y[:,:,1])
    Cr_jpeg = JPEG(y[:,:,2])
    X_ycbcr_jpeg = np.dstack((Y_jpeg, Cb_jpeg, Cr_jpeg))
    X_rgb_jpeg = ycbcr2rgb(X_ycbcr_jpeg)
    X_rgb_jpeg = np.clip(X_rgb_jpeg, 0, 255)
    plt.subplot(121).imshow(X.astype(np.uint8))
    plt.subplot(122).imshow(X_rgb_jpeg.astype(np.uint8))
    plt.axis("off")
    plt.show()

def ex3(X,is_rgb):
    target_MSE = 100 #setam MSE tinta.
    if is_rgb:
        ycbcr = rgb2ycbcr(X1.astype(float))
        Y = ycbcr[:, :, 0]
        Y_jpeg, best_alpha = compress_to_MSE(Y, target_MSE)

        X_ycbcr_jpeg = np.dstack((Y_jpeg, ycbcr[:, :, 1], ycbcr[:, :, 2]))
        X_rgb_jpeg = ycbcr2rgb(X_ycbcr_jpeg)
        X_jpeg = np.clip(X_rgb_jpeg, 0, 255)
    else:
        X_jpeg, best_alpha = compress_to_MSE(X, target_MSE)
    #print(f"Best alpha for target MSE {target_MSE}: {best_alpha}")

    plt.subplot(121).imshow(X.astype(np.uint8), cmap='gray', vmin=X.min(), vmax=X.max())
    plt.subplot(122).imshow(X_jpeg.astype(np.uint8), cmap='gray', vmin=X_jpeg.min(), vmax=X_jpeg.max())
    plt.title(f"Target MSE ={target_MSE} Alpha = {best_alpha}")
    plt.axis("off")
    plt.show()

def compress_frame_rgb(frame, alpha=10):
    ycbcr = rgb2ycbcr(frame)

    H, W, _ = ycbcr.shape
    #facem frameul sa fie multiplu de 8 pt a putea comprima JPEG corect
    H8 = (H //8) * 8
    W8 = (W //8) * 8
    ycbcr = ycbcr[:H8, :W8]
    Y = ycbcr[:, :, 0]
    Y_jpeg = JPEG(Y, alpha=alpha)

    frame_ycbcr_jpeg = np.dstack((
        Y_jpeg,
        ycbcr[:, :, 1],
        ycbcr[:, :, 2]
    ))

    frame_rgb_jpeg = ycbcr2rgb(frame_ycbcr_jpeg)
    return np.clip(frame_rgb_jpeg, 0, 255)

def ex4():
    MAX_FRAMES = 900 # 30 fps * 30 sec
    reader = iio.imread("original.mp4")

    frames = []
    compressed_frames = []

    for i, frame in enumerate(reader):
        if i >= MAX_FRAMES:
            break

        frame = frame.astype(float)
        compressed_frame = compress_frame_rgb(frame, alpha=1)

        frames.append(frame.astype(np.uint8))
        compressed_frames.append(compressed_frame.astype(np.uint8))


    iio.imwrite("compressed_video.mp4", compressed_frames, fps=30)


#ex1()
#ex2(X1)
ex3(X, is_rgb=False)
#ex3(X1, is_rgb=True)
#ex4()