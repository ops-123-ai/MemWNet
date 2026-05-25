import torch
from torchvision import datasets, transforms
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import math
try:
    from skimage.metrics import structural_similarity as _ssim
except ImportError:
    from skimage.measure import compare_ssim as _ssim
from skimage.registration import phase_cross_correlation as register_translation
from scipy import signal


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def ssim(img1, img2, cs_map=False):
    if isinstance(img1, torch.Tensor):
        # print('Tensor!!')
        img1 = img1.squeeze()
        img2 = img2.squeeze()
        img1 = img1.cpu().numpy()
        img2 = img2.cpu().numpy()
        # print('img1.shape:',img1.shape)
        if np.max(img2) < 2:
            # print('data range:[0,1]')
            img1 = img1 * 255
            img2 = img2 * 255

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
        return ssim.mean()


def plt_img(to_plot, nrow, ncol, show=False, save=False, file_path=None):
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol*8, nrow*8))
    for i in range(nrow*ncol):
        j = i//ncol
        k = i % ncol
        axs[j, k].imshow(to_plot[i][0], cmap='gray')
        axs[j, k].axis('off')
    plt.tight_layout()
    if show:
        plt.show()
    if save:
        plt.savefig(file_path)
    plt.close()

def A(x):
    '''
    x : batch data shape: (batch, 1, imsize, imsize)
    result : OverSampled Fourier Transform of x shape:(batch, 1, 2*imsize, 2*imsize)
    '''
    imsize1 = x.shape[2]
    imsize2 = imsize1*2
    pad_num = int((imsize2-imsize1)/2)
    pad = torch.nn.ZeroPad2d((pad_num, pad_num, pad_num, pad_num))
    x = pad(x)
    oversamp_x_fft = torch.fft.fft2(x)*(1/imsize2)*(imsize1/imsize2)

    return oversamp_x_fft


def AT(x):
    '''
    x: OverSampled Fourier Transform of original data shape:(batch, 1, 2*imsize, 2*imsize)
    result:Inverse Fourier Transform and then left multiply (batch,1,imsize,imsize)
    '''
    imsize2 = x.shape[2]
    imsize1 = int(imsize2/2)
    crop_num = int((imsize2-imsize1)/2)
    ifftx = torch.real(torch.fft.ifft2(x)*imsize2*(imsize1/imsize2))
    oversampMTx = ifftx[:, :, crop_num:-crop_num, crop_num:-crop_num]
    return oversampMTx


def A_CDP(x, SamplingRate, mask, device):
    '''
    CDP measurement matrix
    x: batch data shape: (batch, 1, imsize, imsize)
    mask: uniform masks shape: (1, SamplingRate, imsize, imsize)
    '''
    imsize = x.shape[2]   # 128
    # batch_size,SamplingRate,imsize,imsize
    x = x.repeat(1, SamplingRate, 1, 1)
    x = mask*x
    # Ax = torch.zeros_like(x).to(device)   # batchsize*4*128*128 complex
    Ax = torch.zeros_like(x,dtype=torch.complex64).to(x.device) #bmask
    for i in range(SamplingRate):
        Ax[:, i, :, :] = torch.fft.fft2(   
            x[:, i, :, :])*(1/imsize)   # 128*128_complex
    return Ax   # batchsize*4*128*128_complex


def At_CDP(Ax, SamplingRate, mask):
    '''
    CDP measurement inverse matrix
    Ax: OverSampled Fourier Transform of original data shape:(batch_size, SamplingRate, imsize, imsize)

    '''
    B, C, imsize1, imsize2 = Ax.shape  # 128
    Atx = torch.zeros_like(Ax)  # batch_size*SamplingRate*128*128
    for i in range(SamplingRate):
        Atx[:, i, :, :] = torch.fft.ifft2(Ax[:, i, :, :])   # 128*128
    mask_ = torch.conj(mask)   # batch_size * SamplingRate * 128 * 128_complex
    # batch_size * 1 * 128 * 128_complex
    Atx = torch.sum(mask_ * Atx, axis=1)*imsize1
    return torch.real(Atx).reshape(B, 1, imsize1, imsize2)  # 128*128


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def cross_correlation(moving, fixed):

    if moving.shape[-1] == 3:
        moving_gray = rgb2gray(moving)
        fixed_gray = rgb2gray(fixed)
    elif moving.shape[-1] == 1:
        moving_gray = moving[..., 0]
        fixed_gray = fixed[..., 0]
    else:
        print("Image channel Error!")

    shift, error, diffphase = register_translation(moving_gray, fixed_gray)
    out = np.roll(moving, -np.array(shift).astype(np.int), axis=(0, 1))
    return out, error


def register_croco(predicted_images, true_images, torch=True):
    pred_reg = np.empty(predicted_images.shape, dtype=predicted_images.dtype)

    for i in range(len(true_images)):
        if torch:
            true_image = true_images[i].transpose(1, 2, 0)
            predicted_image = predicted_images[i].transpose(1, 2, 0)
        else:
            true_image = true_images[i]
            predicted_image = predicted_images[i]

        shift_predict, shift_error = cross_correlation(
            predicted_image, true_image)
        rotshift_predict, rotshift_error = cross_correlation(
            np.rot90(predicted_image, k=2, axes=(0, 1)), true_image)

        if torch:
            pred_reg[i] = shift_predict.transpose(
                2, 0, 1) if shift_error <= rotshift_error else rotshift_predict.transpose(2, 0, 1)
        else:
            pred_reg[i] = shift_predict if shift_error <= rotshift_error else rotshift_predict

    return pred_reg


class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        data = torch.tensor(self.data[index, ...])
        return data.float()

    def __len__(self):
        return self.len


class ImageOnly(Dataset):

    def __init__(self, orig_dataset):
        self.orig_dataset = orig_dataset

    def __len__(self):
        return len(self.orig_dataset)

    def __getitem__(self, idx):
        return self.orig_dataset[idx][0]


def Poisson_noise_torch(Mx, alpha, device):
    norm = torch.abs(Mx)  # |Ax|
    alpha = torch.Tensor([alpha]).to(device)  # noise level
    B, C, w, h = norm.shape
    intensity_noise = alpha/255*norm*torch.randn(B, C, w, h).to(device)
    y = norm ** 2 + intensity_noise
    y = y*(y > 0)
    y = torch.sqrt(y+1e-5)
    return y
