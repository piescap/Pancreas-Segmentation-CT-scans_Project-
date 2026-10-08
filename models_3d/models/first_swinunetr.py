
import os
import glob
import random
import numpy as np
import torch

from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    AsDiscrete,
)
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
import nibabel
# ==========================================
# 1. GLOBAL SETTINGS
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_size = (96,96,96)
max_epochs = 50       
val_interval = 1        

# PATHS
base_dir = "/home/3175552/SecondCNN/dataset_split_1303" 
pretrained_path = "/home/3175552/3D_UNET/model_swinvit.pt" 
best_model_path = "/home/3175552/3D_UNET/best_swinunetr_pancreas.pt" # Semplificato e in .pt

# ==========================================
# 2. PREPARATION OF DATA
# ==========================================
train_images = sorted(glob.glob(os.path.join(base_dir, "train", "ct", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(base_dir, "train", "mask", "*.nii.gz")))

val_images = sorted(glob.glob(os.path.join(base_dir, "val", "ct", "*.nii.gz")))
val_labels = sorted(glob.glob(os.path.join(base_dir, "val", "mask", "*.nii.gz")))

assert len(train_images) > 0, f"Nessuna immagine di training trovata in {base_dir}/train/ct!"
assert len(val_images) > 0, f"Nessuna immagine di validazione trovata in {base_dir}/val/ct!"

train_files = [{"image": i, "label": l} for i, l in zip(train_images, train_labels)]
val_files = [{"image": i, "label": l} for i, l in zip(val_images, val_labels)]

# ==========================================
# 3. TRANSFORMS E DATA LOADERS
# ==========================================
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=patch_size,
        pos=2,
        neg=1,
        num_samples=4,
        image_key="image",
        image_threshold=0,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
])

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5, num_workers=1)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.5, num_workers=1)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

# ==========================================
# 4. MODEL AND PRE-TRAINED WEIGHTS
# ==========================================
model = SwinUNETR(
    in_channels=1,
    out_channels=1,
    feature_size=48,
    use_checkpoint=True, 
).to(device)

if not os.path.exists(pretrained_path):
    raise FileNotFoundError(
        f"\n[ERRORE] Il file pre-addestrato non si trova in: {pretrained_path}\n"
        "Carica 'model_swinvit.pt' nella cartella 'pretrained' prima di avviare."
    )

def load_pretrained_swinunetr_backbone(model, ckpt):
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model_dict = model.state_dict()
    filtered = {}

    for k, v in state_dict.items():
        new_k = k
        for prefix in ["module.", "swinViT.", "backbone."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]

        if new_k in model_dict and model_dict[new_k].shape == v.shape:
            filtered[new_k] = v
            continue

        candidate = "swinViT." + new_k
        if candidate in model_dict and model_dict[candidate].shape == v.shape:
            filtered[candidate] = v
            continue

    model.load_state_dict(filtered, strict=False)

print(f"Caricamento pesi locali da {pretrained_path}...")
ckpt = torch.load(pretrained_path, map_location=device)
load_pretrained_swinunetr_backbone(model, ckpt)
print("Pesi iniziali caricati con successo!")

# ==========================================
# 5. TRAINING LOOP
# ==========================================
loss_function = DiceCELoss(sigmoid=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

dice_metric = DiceMetric(include_background=True, reduction="mean")
post_pred = AsDiscrete(threshold=0.5)
post_label = AsDiscrete(threshold=0.5)

best_metric = -1

if __name__ == "__main__":
    print(f"Inizio addestramento per {max_epochs} epoche sul device: {device}...")
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= step
        print(f"Epoch {epoch + 1}/{max_epochs} - Train Loss: {epoch_loss:.4f}")

        # Validation and model saving
        if (epoch + 1) % val_interval == 0:
            model.eval()
            dice_metric.reset()

            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs = val_data["image"].to(device)
                    val_labels = val_data["label"].to(device)

                    val_outputs = sliding_window_inference(
                        val_inputs,
                        roi_size=patch_size,
                        sw_batch_size=2, 
                        predictor=model,
                        overlap=0.5,
                    )

                    val_outputs = [post_pred(i) for i in decollate_batch(torch.sigmoid(val_outputs))]
                    val_labels_list = [post_label(i) for i in decollate_batch(val_labels)]

                    dice_metric(y_pred=val_outputs, y=val_labels_list)

                metric = dice_metric.aggregate().item()
                print(f"Epoch {epoch + 1}/{max_epochs} - Val Mean Dice: {metric:.4f}")


                if metric > best_metric:
                    best_metric = metric
                    torch.save(model.state_dict(), best_model_path)
                    print(f"--> Nuovo miglior modello salvato in: {best_model_path} (Dice: {best_metric:.4f})")

    print(f"Training completato. Il miglior Dice Score è stato: {best_metric:.4f}.")