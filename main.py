import math
import os
import random

import torch
import torch.nn as nn

from torchvision.transforms import v2, functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.optim import AdamW

from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pytorch_msssim import ms_ssim

from data_loading import CocoDenoisingDataset
from architecture import NAFNet
from saltnpepper_transform import BatchSaltAndPepper

import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo


img_size = 256
rand_state = 44
num_workers = 12

def curr_time():
    return datetime.now(ZoneInfo('Europe/Kiev'))


def printshare(msg, logfile="training_log.txt"):
    print(msg)

    with open(logfile, "a") as f:
        print(msg, file=f)


def cosannealing_decay_warmup(warmup_steps, T_0, T_mult, decay_factor, base_lr, eta_min):
    # returns the func that performs all the calculations.
    # useful for keeping all the params in one place = scheduler def.
    def lr_lambda(epoch): #0-based epoch
        if epoch < warmup_steps:
            return base_lr * ((epoch + 1) / warmup_steps)

        annealing_step = epoch - warmup_steps

        # calculating which cycle (zero-based) are we in,
        # current cycle length (T_current) and position inside the cycle (t)
        if T_mult == 1:
            cycle = annealing_step // T_0
            t = annealing_step % T_0
            T_current = T_0

        else:
            # fast log-based computation
            cycle = int(math.log((annealing_step * (T_mult - 1)) / T_0 + 1, T_mult))
            sum_steps_of_previous_cycles = T_0 * (T_mult ** cycle - 1) // (T_mult - 1)
            t = annealing_step - sum_steps_of_previous_cycles
            T_current = T_0 * (T_mult ** cycle)


        # enable decay
        eta_max = base_lr * (decay_factor ** cycle)

        # cosine schedule between (eta_min, max_lr]
        lr = eta_min + 0.5 * (eta_max-eta_min) * (1 + math.cos(math.pi * t / T_current))
        return lr/base_lr

    return lr_lambda






def perform_training(net,
                     training_set,
                     validation_set,
                     epochs, w_decay, batch_size, sub_batch_size,
                     lr, lr_lambda: cosannealing_decay_warmup,
                     pretrained: bool | str = False):

    assert batch_size % sub_batch_size == 0 #screws up gradient accumulation otherwise

    printshare("training preparation...")

    scaler = torch.amp.GradScaler('cuda')

    train_loader = DataLoader(training_set, batch_size=sub_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_set, batch_size=sub_batch_size, shuffle=True, num_workers=num_workers)

    #========= loading the checkpoint and preparing optimizers =========

    criterion = nn.L1Loss()
    optimizer = AdamW(
        params=filter(lambda p: p.requires_grad, net.parameters()),
        lr=lr, weight_decay=w_decay)
        #[
        #    {"params": net.features[-2].parameters()},  # last residual block
        #    {"params": net.features[-1].parameters()},  # last conv
        #    {"params": net.classifier.parameters()}  # classifier
        #],

    #used LambdaLR to implement CosineAnnealing with warm restarts and decay.
    #yup, we need the base_lr to be passed in, cause it looks like this is the safest way.
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )

    #scheduler = CosineAnnealingLR(
    #    optimizer=optimizer,
    #    T_max=50,
    #    eta_min=1e-8,
    #)

    curr_epoch = 0
    if isinstance(pretrained, str):
        printshare("Loading pretrained model, optimizer & scheduler state dicts...")
        checkpoint = torch.load(pretrained)
        mid_se_keys = ["mid_se.fc.0.weight", "mid_se.fc.0.bias", "mid_se.fc.2.weight", "mid_se.fc.2.bias"]

        if 'model' not in checkpoint:
            missing, unexpected = net.load_state_dict(checkpoint, strict=False)
            printshare("got no optimizer & scheduler state dicts. model state dict set up successfully.")

        else:
            missing, unexpected = net.load_state_dict(checkpoint['model'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['weight_decay'] = w_decay

            #scheduler.load_state_dict(checkpoint["scheduler"])
            scheduler.last_epoch = checkpoint['epoch']
            curr_epoch = checkpoint['epoch'] + 1

            printshare("all the dicts set up successfully.")


        printshare(f"[DEBUG] model missing statedict vals: {missing};")
        printshare(f"[DEBUG] model unexpected statedict vals: {unexpected}")

    #manual testing cycle
    while(True):

        input_img, target_img = training_set[random.randint(0, len(training_set)-1)]
        transform = v2.ToPILImage()
        inp_img = transform(input_img)
        target = transform(target_img)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(inp_img)
        axes[0].set_title("Noisy input")
        axes[0].axis("off")

        axes[1].imshow(target)
        axes[1].set_title("Target")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/stats", exist_ok=True)
    printshare("done.")

    #========== training itself ==========
    while curr_epoch < epochs:
        printshare(f"[{curr_time().strftime('%Y-%m-%d %H:%M:%S')}] epoch {curr_epoch + 1}/{epochs} processing...")
        output_train_msssim, input_train_msssim, train_loss = perform_training_epoch(
            net=net,
            full_batch_size=batch_size, sub_batch_size=sub_batch_size,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler
        )

        printshare(f"training done. input ms-ssim: {round(input_train_msssim, 3)}%;"+
                   f" output ms-ssim: {round(output_train_msssim, 3)}%")


        printshare(f"[{curr_time().strftime('%Y-%m-%d %H:%M:%S')}] processing validation phase...")
        output_val_msssim, input_val_msssim, val_loss = perform_validation_epoch(
            net=net,
            val_loader=val_loader,
            criterion=criterion,
            scaler=scaler
        )

        printshare(f"validation done. input ms-ssim: {round(input_val_msssim, 3)}%;" +
                   f" output ms-ssim: {round(output_val_msssim, 3)}%")

        torch.save({ # model
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': curr_epoch,

        }, f'checkpoints/ep_{curr_epoch+1}_ts_{round(output_train_msssim, 1)}_vs_{round(output_val_msssim, 1)}_model.pth')

        torch.save({ # stats
            'epoch': curr_epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'input_train_msssim': input_train_msssim,
            'output_train_msssim': output_train_msssim,
            'input_val_msssim': input_val_msssim,
            'output_val_msssim': output_val_msssim
        },
            f'checkpoints/stats/ep_{curr_epoch+1}_ts_{round(output_train_msssim, 1)}_vs_{round(output_val_msssim, 1)}_stats.pth')

        curr_epoch += 1

    printshare(f"[{curr_time().strftime('%Y-%m-%d %H:%M:%S')}] training successfully finished.")
    return net


def perform_training_epoch(net, full_batch_size, sub_batch_size,
                           train_loader, criterion, optimizer, scheduler,
                           scaler):
    batch_losses = []
    model_output_msssim_vals = []
    input_msssim_vals = []

    net.train()

    accum_steps = math.ceil(full_batch_size / sub_batch_size)
    optimizer.zero_grad()

    for i, (input_imgs, target_imgs) in enumerate(train_loader):
        input_imgs, target_imgs = input_imgs.cuda(), target_imgs.cuda()

        with torch.amp.autocast('cuda'):
            outputs = net(input_imgs)
            outputs = torch.clamp(outputs, 0.0, 1.0)

            loss = criterion(outputs, target_imgs)
            loss = loss / accum_steps

        # MS-SSIM usually expects Float32
        with torch.no_grad():
            outputs_f32 = outputs.detach().float()

            model_output_batch_msssim = ms_ssim(outputs_f32,
                                                target_imgs,
                                                data_range=1.0, size_average=True)
            model_output_msssim_vals.append(model_output_batch_msssim.item())

            input_batch_msssim = ms_ssim(input_imgs,
                                         target_imgs,
                                         data_range=1.0, size_average=True)
            input_msssim_vals.append(input_batch_msssim.item())


        scaler.scale(loss).backward()

        batch_losses.append(loss.item() * accum_steps)

        if (i + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    scheduler.step()

    epoch_loss = sum(batch_losses) / len(batch_losses)

    avg_model_msssim = sum(model_output_msssim_vals) / len(model_output_msssim_vals) if len(
        model_output_msssim_vals) > 0 else 0
    avg_input_msssim = sum(input_msssim_vals) / len(input_msssim_vals) if len(input_msssim_vals) > 0 else 0

    return avg_model_msssim, avg_input_msssim, epoch_loss


def perform_validation_epoch(net, val_loader, criterion):
    net.eval()
    with torch.no_grad():
        batch_losses = []
        model_output_msssim_vals = []
        input_msssim_vals = []

        for input_imgs, target_imgs in val_loader:
            input_imgs, target_imgs = input_imgs.cuda(), target_imgs.cuda()

            with torch.amp.autocast('cuda'):
                outputs = net(input_imgs)
                outputs = torch.clamp(outputs, 0.0, 1.0)

                loss_val = criterion(outputs, target_imgs)

            outputs_f32 = outputs.float()

            model_batch_msssim = ms_ssim(outputs_f32,
                                         target_imgs,
                                         data_range=1.0, size_average=True)

            input_batch_msssim = ms_ssim(input_imgs,
                                         target_imgs,
                                         data_range=1.0, size_average=True)

            model_output_msssim_vals.append(model_batch_msssim.item())
            input_msssim_vals.append(input_batch_msssim.item())
            batch_losses.append(loss_val.item())

        epoch_loss = sum(batch_losses) / len(batch_losses)

        avg_model_msssim = sum(model_output_msssim_vals) / len(model_output_msssim_vals) if len(
            model_output_msssim_vals) > 0 else 0
        avg_input_msssim = sum(input_msssim_vals) / len(input_msssim_vals) if len(input_msssim_vals) > 0 else 0

        return avg_model_msssim, avg_input_msssim, epoch_loss

def perform_testing(net, test_set, bs=4, weights_file=""):
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=num_workers)
    if isinstance(weights_file, str):
        printshare("Loading pretrained model, optimizer & scheduler state dicts...")
        checkpoint = torch.load(weights_file)

        if 'model' not in checkpoint:
            _, _ = net.load_state_dict(checkpoint, strict=False)
            printshare("got no optimizer & scheduler state dicts. model state dict set up successfully.")

        else:
            _, _ = net.load_state_dict(checkpoint['model'], strict=False)

            printshare("all the dicts set up successfully.")

    net.eval()
    with torch.no_grad():

        for input_imgs, target_imgs in test_loader:
            input_imgs, target_imgs = input_imgs.cuda(), target_imgs.cuda()

            outputs = net(input_imgs)
            outputs = torch.clamp(outputs, 0.0, 1.0)

            outputs_f32 = outputs.float()

            for input, output, target in zip(input_imgs, outputs_f32, target_imgs):
                transform = v2.ToPILImage()
                inp = transform(input.cpu())
                out = transform(output.cpu())
                tar = transform(target.cpu())

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(inp)
                axes[0].set_title("Noisy input")
                axes[0].axis("off")

                axes[1].imshow(out)
                axes[1].set_title("Output")
                axes[1].axis("off")

                axes[2].imshow(tar)
                axes[2].set_title("Target")
                axes[2].axis("off")

                plt.tight_layout()
                plt.show()



        return 0, 0, 0



def custom_loader(path):
    return Image.open(path, formats=["JPEG"])




if __name__ == '__main__':

    net = NAFNet()
    net.cuda(0)

    noise_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(img_size, img_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianNoise(mean=0, sigma=0.08),
        BatchSaltAndPepper(salt_prob=0.05, pepper_prob=0.05),
        v2.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5))
    ])

    base_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(img_size, img_size)),
        v2.ToDtype(torch.float32, scale=True),
    ])

    train_set = CocoDenoisingDataset("data/train", input_img_transform=noise_transform,
                                     target_img_transform=base_transform)
    val_set = CocoDenoisingDataset("data/val", input_img_transform=noise_transform,
                                     target_img_transform=base_transform)
    test_set = CocoDenoisingDataset("data/test", input_img_transform=noise_transform,
                                     target_img_transform=base_transform)

    #perform_training(net, train_set, val_set,
    #                 epochs=600, w_decay=1e-4, batch_size=64, sub_batch_size=4,
    #                 lr=1e-3, lr_lambda=cosannealing_decay_warmup(
    #                   warmup_steps=0, T_0=10, T_mult=1.1, decay_factor=0.9, base_lr=1e-3, eta_min=1e-8),
    #                 pretrained='checkpoints/ep_2_ts_1.0_vs_1.0_model.pth')

    perform_testing(net, test_set, weights_file='checkpoints/ep_11_ts_95.1_vs_95.1_model.pth')