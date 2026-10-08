import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from monai.utils import set_determinism

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandCropByPosNegLabeld,
    AsDiscrete,
    NormalizeIntensityd,
    Activations, # ADDED: Necessary to transform logits to probabilities
    RandFlipd,           # ADDED: Per rimpiazzare la tua funzione custom
    RandRotated,         # ADDED: Per rimpiazzare la tua funzione custom
    RandScaleIntensityd  # ADDED: Per rimpiazzare la tua funzione custom
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)

CONFIG = {
    "seed": 42,
    "data_dir": "/home/3175552/SecondCNN/dataset_split_1303",
    "output_dir": "/home/3175552/UNET_MATI",
    "train_dir": "train",
    "val_dir": "val",
    "test_dir": "test",
    "batch_size": 2,
    "num_workers": 1,
    "learning_rate": 1e-4, 
    "max_epochs": 100,
    "val_interval": 1,
    "patch_size": (96, 96, 96),
    "in_channels": 1,
    "out_channels": 1,
    "channels": (16, 32, 64, 128, 256),
    "strides": (2, 2, 2, 2),
}

# --- 0.5 BCE + 0.5 DICE ---
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)
        return (0.5 * bce_loss) + (0.5 * dice_loss)

def prepare_paths(config):
    data_root = Path(config["data_dir"])
    output_root = Path(config["output_dir"])
    paths = {
        "train_dir": data_root / config["train_dir"],
        "val_dir": data_root / config["val_dir"],
        "test_dir": data_root / config["test_dir"],
        "checkpoints_dir": output_root / "checkpoints",
        "logs_dir": output_root / "logs",
    }
    paths["checkpoints_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    return paths

def get_case_dicts(split_dir):
    split_dir = Path(split_dir)
    ct_dir = split_dir / "ct"
    mask_dir = split_dir / "mask"
    ct_files = sorted(ct_dir.glob("*.nii.gz"))
    mask_files = sorted(mask_dir.glob("*.nii.gz"))
    cases = []
    for ct_path, mask_path in zip(ct_files, mask_files):
        cases.append({"image": str(ct_path), "label": str(mask_path)})
    return cases

def get_baseline_transforms(config):
    # The Data Augmentation is now handle directly by MONAI (much faster and safer in terms of randomness and reproducibility)
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config["patch_size"],
            pos=1, neg=1, num_samples=2,
            image_key="image", image_threshold=0,
        ),
        # --- AUGMENTATION NATIVA 3D ---
        RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
        RandRotated(keys=["image", "label"], range_x=0.26, range_y=0.26, range_z=0.26, mode=["bilinear", "nearest"], prob=0.3),
        RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.5),
        # ------------------------------
        EnsureTyped(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"]),
    ])
    return train_transforms, val_transforms

def train_one_epoch(model, train_loader, optimizer, scheduler, loss_function, device):
    model.train()
    epoch_loss = 0.0
    for batch_data in train_loader:
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

# ADDED: Now accepts the loss_function to calculate the Validation Loss
def validate(model, val_loader, loss_function, dice_metric, post_pred, post_label, config, device):
    model.eval()
    dice_metric.reset()
    val_loss_total = 0.0
    
    with torch.no_grad():
        for val_data in val_loader:
            val_images = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)
            
            val_outputs = sliding_window_inference(
                inputs=val_images,
                roi_size=config["patch_size"],
                sw_batch_size=2, 
                predictor=model,
            )
            
            # 1. Calculation of Validation Loss (on raw logits)
            loss = loss_function(val_outputs, val_labels)
            val_loss_total += loss.item()

            # 2. Post-processing for Dice Metric
            val_outputs_list = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels_list = [post_label(i) for i in decollate_batch(val_labels)]
            dice_metric(y_pred=val_outputs_list, y=val_labels_list)
            
    avg_val_loss = val_loss_total / len(val_loader)
    avg_val_dice = dice_metric.aggregate().item()
    return avg_val_loss, avg_val_dice

def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = prepare_paths(CONFIG)
    
    train_files = get_case_dicts(paths["train_dir"])
    val_files = get_case_dicts(paths["val_dir"])
    train_transforms, val_transforms = get_baseline_transforms(CONFIG)

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=CONFIG["num_workers"])
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=CONFIG["num_workers"])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=CONFIG["num_workers"])

    model = UNet(
        spatial_dims=3,
        in_channels=CONFIG["in_channels"],
        out_channels=CONFIG["out_channels"],
        channels=CONFIG["channels"],
        strides=CONFIG["strides"],
        num_res_units=2,
    ).to(device)

    loss_function = CombinedLoss() 
    dice_metric = DiceMetric(include_background=True, reduction="mean")

  
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    LR_MAX = CONFIG["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX / 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR_MAX,
        steps_per_epoch=len(train_loader),
        epochs=CONFIG["max_epochs"],
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=1e4
    )

    best_metric = -1
    for epoch in range(CONFIG["max_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_function, device)
        
        if (epoch + 1) % CONFIG["val_interval"] == 0:
            # Ora validate ritorna sia la loss che il dice
            val_loss, val_dice = validate(model, val_loader, loss_function, dice_metric, post_pred, post_label, CONFIG, device)
            
            print(f"Epoch {epoch+1}/{CONFIG['max_epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

            if val_dice > best_metric:
                best_metric = val_dice
                save_path = paths["checkpoints_dir"] / "baseline_best_model.pt"
                torch.save(model.state_dict(), save_path)
                print(f"--- Nuovo record! Modello salvato: {save_path} ---")

    print(f"Training completato. Miglior Dice Validation: {best_metric:.4f}")

if __name__ == "__main__":
    main()