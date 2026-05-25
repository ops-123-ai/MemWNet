import os
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import time
import random
import pickle
import torch
import torch.utils.data
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from tqdm import tqdm

import loader
from models.common import config
from models.networks import MemWNet
from utils import A_CDP, Poisson_noise_torch
from test import testing


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        print(f"checking paths, mkdir: {path}")


def main():
    check_path(config.para.save_path)
    check_path(config.para.folder)
    set_seed(99999)

    net = MemWNet.unfold_net(num_stages=config.para.stage).train().to(config.para.device)

    optimizer = optim.AdamW(filter(lambda x: x.requires_grad, net.parameters()), lr=config.para.lr)
    scheduler = LS.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    if os.path.exists(config.para.my_state_dict):
        if not torch.cuda.is_available():
            raise Exception("No GPU.")
        ckpt = torch.load(config.para.my_state_dict, map_location=config.para.device)
        net.load_state_dict(ckpt, strict=False)
        info = torch.load(config.para.my_info, map_location=config.para.device, weights_only=False)
        start_epoch = info["epoch"]
        current_best = info["res"]
        print(f"Loaded trained model of epoch {start_epoch}, res: {current_best}.")
    else:
        start_epoch = 1
        current_best = 0
        print("No saved model, start epoch = 1.")

    print("Data loading...")
    train_set = loader.TrainDatasetFromFolder('../dataset/train/DIV2K_BSD400', block_size=config.para.patch_size)
    dataset_train = torch.utils.data.DataLoader(
        dataset=train_set, num_workers=1, batch_size=config.para.batch_size, shuffle=True, pin_memory=False)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    os.makedirs(config.para.matrix_dir, exist_ok=True)

    over_all_time = time.time()
    for epoch in range(start_epoch, 201):
        ave_loss = 0.0
        epoch_loss = 0.0
        print(f"Please note:    Lr: {optimizer.param_groups[0]['lr']}.\n")
        dic = {"epoch": epoch, "device": config.para.device, "rate": config.para.rate}

        for idx, xi in enumerate(tqdm(dataset_train, desc="Now training: ", postfix=dic)):
            alpha = random.choice([9, 27, 81])

            with torch.cuda.amp.autocast(enabled=True):
                xi = xi.to(config.para.device)

                Mask_data_Name = f'./{config.para.matrix_dir}/bipolar_mask_{config.para.rate}_{config.para.patch_size}_train.p'
                if os.path.exists(Mask_data_Name):
                    Mask_data = pickle.load(open(Mask_data_Name, 'rb'))
                else:
                    # uniform mask
                    # Mask_data = torch.exp(
                    #     1j*2*torch.pi*torch.rand(1, config.para.rate, config.para.patch_size, config.para.patch_size)).to(device)
                    # pickle.dump(Mask_data, open(Mask_data_Name, 'wb'))
                    # bipolar mask
                    probability = torch.ones(1, config.para.rate, config.para.patch_size, config.para.patch_size) * 0.5
                    Mask_data = (torch.bernoulli(probability) * 2 - 1).to(config.para.device)
                    pickle.dump(Mask_data, open(Mask_data_Name, 'wb'))
                mask = Mask_data.to(config.para.device)
                b = Poisson_noise_torch(
                    A_CDP(xi, SamplingRate=config.para.rate, mask=mask, device=config.para.device),
                    alpha=alpha, device=config.para.device)

                optimizer.zero_grad()
                initial_data = torch.ones_like(xi)
                xo = net(initial_data, b, config.para.rate, mask, config.para.device)
                batch_loss = torch.mean(torch.pow(xo - xi, 2)).to(config.para.device)
                epoch_loss += batch_loss.item()
                ave_loss = (ave_loss * idx + batch_loss.item()) / (idx + 1)

            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if idx % 10 == 0:
                tqdm.write(f"\r[{config.para.batch_size * (idx + 1):5}/{len(dataset_train) * config.para.batch_size:5}], "
                           f"Loss: [{batch_loss.item():8.6f}], AveLoss: [{ave_loss:8.6f}]")

        avg_loss = epoch_loss / len(dataset_train)
        print(f"\n=> Epoch of {epoch:2}, Epoch Loss: [{avg_loss:8.6f}]")

        if epoch == 1:
            if not os.path.isfile(config.para.my_log):
                with open(config.para.my_log, 'w') as f:
                    f.write("=" * 120 + "\n")
            with open(config.para.my_log, 'r+') as f:
                old = f.read()
                f.seek(0)
                f.write(f"\nAbove is ??? test. Note: None.\n" + "=" * 120 + "\n")
                f.write(old)

        test_interval = 1
        with torch.no_grad():
            if epoch % test_interval == 0 or epoch == 1:
                p, s = testing(net.eval(), val=True, save_img=True)
                print(f"{p:5.3f}")
                if p > current_best:
                    epoch_info = {"epoch": epoch, "res": p}
                    torch.save(net.state_dict(), config.para.my_state_dict)
                    torch.save(epoch_info, config.para.my_info)
                    print("Check point saved\n")
                    current_best = p
                    with open(config.para.my_log, 'r+') as f:
                        old = f.read()
                        f.seek(0)
                        f.write(f"Epoch {epoch}, Loss of train {round(avg_loss, 6)}, "
                                f"Res {round(current_best, 2)}, {round(s, 4)}\n")
                        f.write(old)
            else:
                print(f"Skipping testing at epoch {epoch} "
                      f"(will test at epoch {((epoch // test_interval) + 1) * test_interval})")
                with open(config.para.my_log, 'a') as f:
                    f.write(f"Epoch {epoch}, Loss of train {round(avg_loss, 6)}, No test\n")

        scheduler.step()
        print(f"Epoch time: {time.time() - over_all_time:.3f}s")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.backends.cudnn.enabled = True
    main()