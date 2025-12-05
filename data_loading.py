import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms.functional as F
class CocoDenoisingDataset(Dataset):
    def __init__(self, root_dir, input_img_transform, target_img_transform, noise_factor=0.2):
        """
        Args:
            root_dir (string): Шлях до папки з картинками.
            img_size (tuple): Розмір, до якого треба ресайзити (h, w).
            noise_factor (float): Сила шуму (від 0.0 до 1.0).
        """
        self.root_dir = root_dir
        self.input_transform = input_img_transform
        self.target_transform = target_img_transform

        # Отримуємо список всіх файлів (фільтруємо тільки картинки)
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Базовий трансформ: Ресайз + перетворення в тензор


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])

        # 1. Завантаження картинки
        # convert('RGB') важливо, бо в COCO є чорно-білі фото, які зламають розмірність тензора
        image = Image.open(img_path).convert('RGB')
        # 2. Створення Target (чиста картинка)
        input_img = self.input_transform(image)
        target_img = self.target_transform(image)

        return input_img, target_img