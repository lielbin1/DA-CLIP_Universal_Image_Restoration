import os
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import sys
import lmdb
import torch.utils.data as data

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass

class LLQDataset(Dataset):
    """
    Dataset to handle images with two types of noise from nested directories.
    """

    def __init__(self, opt):
        """
        Args:
            opt (dict): Configuration options.
        """
        super().__init__()
        self.opt = opt
        self.root_dir = opt["dataroot_LQ"]
        self.patch_size = opt["LR_size"]
        self.phase = opt["phase"]
        self.data = []

        # Traverse directories to gather file paths and noise labels
        for noise1 in os.listdir(self.root_dir):
            noise1_path = os.path.join(self.root_dir, noise1)
            if os.path.isdir(noise1_path):
                for noise2 in os.listdir(noise1_path):
                    noise2_path = os.path.join(noise1_path, noise2)
                    if os.path.isdir(noise2_path):
                        for img_name in os.listdir(noise2_path):
                            if img_name.endswith((".png", ".jpg", ".jpeg")):
                                self.data.append({
                                    "path": os.path.join(noise2_path, img_name),
                                    "noise1": noise1,
                                    "noise2": noise2
                                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = sample["path"]
        noise1 = sample["noise1"]
        noise2 = sample["noise2"]

        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Random cropping during training
        if self.phase == "train":
            h, w, _ = img.shape
            top = random.randint(0, max(0, h - self.patch_size))
            left = random.randint(0, max(0, w - self.patch_size))
            img = img[top:top + self.patch_size, left:left + self.patch_size]

            # Augmentation
            img = util.augment(
                img,
                self.opt["use_flip"],
                self.opt["use_rot"],
                self.opt["mode"],
            )

        # Convert to RGB and normalize
        if self.opt["color"]:
            img = util.channel_convert(img.shape[2], self.opt["color"], [img])[0]

        # Transform for CLIP
        lq4clip = util.clip_transform(img)

        # Convert to tensor
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()

        return {"LQ": img, "LQ_clip": lq4clip, "noise1": noise1, "noise2": noise2, "LQ_path": img_path}
