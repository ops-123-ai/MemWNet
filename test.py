import os
import torch
import pickle
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from models.common import config
import csv
from models.networks import MemWNet
from utils import A, AT, A_CDP, At_CDP, Poisson_noise_torch


def save_log(recon_root, name_dataset, name_image, psnr, ssim, rate, consecutive=True):
    if not os.path.isfile(f"{recon_root}/Res_{name_dataset}_{rate}.txt"):
        log = open(f"{recon_root}/Res_{name_dataset}_{rate}.txt", 'w')
        log.write("=" * 120 + "\n")
        log.close()
    log = open(f"{recon_root}/Res_{name_dataset}_{rate}.txt", 'r+')
    if consecutive:
        old = log.read()
        log.seek(0)
        log.write(old)
    log.write(
        f"Res {name_image}: PSNR, {round(psnr, 2)}, SSIM, {round(ssim, 4)}\n")
    log.close()


def imread_CS_py(Iorg):
    block_size = config.para.patch_size
    [row, col] = Iorg.shape
    if np.mod(row, block_size) == 0:
        row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)
    if np.mod(col, block_size) == 0:
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def testing(network, val, device=config.para.device, save_best=True):
    recon_root = f"./bipolar_test/{config.para.rate}_{config.para.alpha}"
    if not os.path.isdir(recon_root):
        os.makedirs(recon_root, exist_ok=True)
    datasets = ["128NT", "128UNT", "BSD68"]
    with torch.no_grad():
        for one_dataset in datasets:
            print(one_dataset + " reconstruction start")
            test_dataset_path = f"../dataset/test/{one_dataset}"
            if os.path.isfile(f"{recon_root}/Res_{one_dataset}_gray_{config.para.rate}.txt"):
                os.remove(f"{recon_root}/Res_{one_dataset}_gray_{config.para.rate}.txt")

            csv_path = (f"{recon_root}/Res_{one_dataset}_gray_"
                        f"rate{config.para.rate}_alpha{config.para.alpha}.csv")
            csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["image", "psnr", "ssim"])

            sum_psnr, sum_ssim = 0., 0.
            img_count = 0
            for _, _, images in os.walk(f"{test_dataset_path}/"):
                for one_image in images:
                    name_image = one_image.split('.')[0]
                    Img = cv.imread(f"{test_dataset_path}/{one_image}", flags=1)
                    Img_yuv = cv.cvtColor(Img, cv.COLOR_BGR2YCrCb)
                    Img_rec_yuv = Img_yuv.copy()
                    Iorg_y = Img_yuv[:, :, 0]
                    [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y)
                    Img_output = Ipad / 255.

                    batch_x = torch.from_numpy(Img_output)
                    batch_x = batch_x.type(torch.FloatTensor)
                    batch_x = batch_x.to(config.para.device)
                    batch_x = batch_x.unsqueeze(0).unsqueeze(0)
                    batch_x = torch.cat(torch.split(batch_x, split_size_or_sections=config.para.patch_size, dim=3), dim=0)
                    batch_x = torch.cat(torch.split(batch_x, split_size_or_sections=config.para.patch_size, dim=2), dim=0)

                    Mask_data_Name = './%s/bipolar_mask_%d_%d_test.p' % (
                        'sampling_matrix', config.para.rate, config.para.patch_size)
                    if os.path.exists(Mask_data_Name):
                        Mask_data = pickle.load(open(Mask_data_Name, 'rb'))
                    else:
                        # uniform mask
                         # Mask_data = torch.exp(
                        #     1j*2*torch.pi*torch.rand(1, config.para.rate, config.para.patch_size, config.para.patch_size)).to(device)
                        # pickle.dump(Mask_data, open(Mask_data_Name, 'wb'))
                        # bipolar mask
                        probability = torch.ones(1, config.para.rate, config.para.patch_size,
                                                 config.para.patch_size) * 0.5
                        Mask_data = (torch.bernoulli(probability) * 2 - 1).to(config.para.device)
                        pickle.dump(Mask_data, open(Mask_data_Name, 'wb'))
                    mask = Mask_data.to(device)

                    b = Poisson_noise_torch(
                        A_CDP(batch_x, SamplingRate=config.para.rate,
                              mask=mask, device=config.para.device),
                        alpha=config.para.alpha, device=config.para.device)
                    initial_data = torch.ones_like(batch_x)
                    x_output = network(initial_data, b, config.para.rate, mask)
                    x_output = torch.cat(torch.split(x_output,
                        split_size_or_sections=1 * col_new // config.para.patch_size, dim=0), dim=2)
                    x_output = torch.cat(torch.split(x_output,
                        split_size_or_sections=1, dim=0), dim=3)
                    x_output = x_output.squeeze(0).squeeze(0)
                    Prediction_value = x_output.cpu().data.numpy()

                    X_rec = Prediction_value[:row, :col]
                    X_rec = np.clip(X_rec, 0, 1) * 255.
                    rec_PSNR = PSNR(X_rec, Iorg.astype(np.float64), data_range=255)
                    rec_SSIM = SSIM(X_rec, Iorg.astype(np.float64), data_range=255)

                    print(f"  {name_image}: PSNR={rec_PSNR:.2f}  SSIM={rec_SSIM:.4f}")

                    csv_writer.writerow([name_image, round(rec_PSNR, 4), round(rec_SSIM, 6)])
                    csv_file.flush()

                    sum_psnr += rec_PSNR
                    sum_ssim += rec_SSIM
                    img_count += 1

                    if save_best:
                        Img_rec_yuv[:, :, 0] = X_rec
                        im_rec_rgb = cv.cvtColor(Img_rec_yuv, cv.COLOR_YCrCb2BGR)
                        im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
                        out_dir = f"{recon_root}/{one_dataset}/{config.para.rate}"
                        os.makedirs(out_dir, exist_ok=True)
                        fname = f"{name_image}_MemWNet_{rec_PSNR:.2f}_{rec_SSIM:.4f}.png"
                        cv.imwrite(f"{out_dir}/{fname}", im_rec_rgb)

                    save_log(recon_root, one_dataset, name_image,
                             rec_PSNR, rec_SSIM, f"gray_{config.para.rate}")
                    del x_output

            if img_count > 0:
                avg_psnr = round(sum_psnr / img_count, 4)
                avg_ssim = round(sum_ssim / img_count, 6)
                csv_writer.writerow(["Average", avg_psnr, avg_ssim])
            csv_file.close()
            if img_count > 0:
                print(f"  {one_dataset} avg: PSNR={sum_psnr/img_count:.2f}  SSIM={sum_ssim/img_count:.4f}")

if __name__ == "__main__":
    my_state_dict = config.para.my_state_dict
    device = config.para.device

    net = MemWNet.unfold_net(num_stages=config.para.stage).eval().to(device)
    if os.path.exists(my_state_dict):
        if torch.cuda.is_available():
            trained_model = torch.load(my_state_dict, map_location=device)
        else:
            raise Exception("No GPU.")
        net.load_state_dict(trained_model)
    else:
        raise FileNotFoundError(f"Missing trained model of rate {config.para.rate}.")
    testing(net, val=False)