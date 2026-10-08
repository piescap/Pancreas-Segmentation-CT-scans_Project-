import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import segmentation_models_pytorch as smp

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────
# READ PATIENT IDS & PATHS
# ─────────────────────────────────────────────
def get_patient_ids(ct_folder):
    ids = []
    for fname in sorted(os.listdir(ct_folder)):
        if not fname.endswith(".nii.gz"):
            continue
        if fname.startswith("pancreas_"):
            pid = fname.replace("pancreas_", "").replace(".nii.gz", "")
        else:
            pid = fname.replace(".nii.gz", "")
        ids.append(pid)
    return ids

def find_ct_path(ct_folder, pid):
    for candidate in [f"pancreas_{pid}.nii.gz", f"{pid}.nii.gz"]:
        p = os.path.join(ct_folder, candidate)
        if os.path.exists(p):
            return p
    return None

def find_mask_path(mask_folder, pid):
    for candidate in [f"label{pid}.nii.gz", f"pancreas_{pid}.nii.gz", f"{pid}.nii.gz"]:
        p = os.path.join(mask_folder, candidate)
        if os.path.exists(p):
            return p
    return None

# ─────────────────────────────────────────────
# DATASET 2D
# ─────────────────────────────────────────────
class Pancreas2DSliceDataset(Dataset):
    def __init__(self, split_dir, pos_fraction=0.5, cache_volumes=True, training=True):
        self.ct_dir = os.path.join(split_dir, "ct")
        self.mask_dir = os.path.join(split_dir, "mask")
        self.pos_fraction = pos_fraction
        self.cache_volumes = cache_volumes
        self.training = training

        self.patient_ids = get_patient_ids(self.ct_dir)
        self.index_all = []
        self.index_pos = []
        self.volumes_ct = {}
        self.volumes_mask = {}

        print(f"Indicizzazione di {len(self.patient_ids)} pazienti da '{split_dir}'...", flush=True)

        for i, pid in enumerate(self.patient_ids):
            ct_path = find_ct_path(self.ct_dir, pid)
            m_path = find_mask_path(self.mask_dir, pid)

            if ct_path is None:
                print(f" [SKIP] CT mancante per {pid}", flush=True)
                continue
            if m_path is None:
                print(f" [SKIP] maschera mancante per {pid}", flush=True)
                continue

            ct_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
            m_array = sitk.GetArrayFromImage(sitk.ReadImage(m_path)).astype(np.uint8)

            if self.cache_volumes:
                self.volumes_ct[pid] = ct_array
                self.volumes_mask[pid] = m_array

            for z in range(ct_array.shape[0]):
                self.index_all.append((pid, z))
                if m_array[z].max() > 0:
                    self.index_pos.append((pid, z))

            if (i + 1) % 10 == 0:
                print(f" > Processati {i+1}/{len(self.patient_ids)} pazienti...", flush=True)

        print(f" Totale slice: {len(self.index_all)} | Slice positive: {len(self.index_pos)}", flush=True)

    def _get_slice(self, pid, z):
        if self.cache_volumes:
            return self.volumes_ct[pid][z], self.volumes_mask[pid][z]

        ct_path = find_ct_path(self.ct_dir, pid)
        m_path = find_mask_path(self.mask_dir, pid)

        ct_slice = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)[z]
        m_slice = sitk.GetArrayFromImage(sitk.ReadImage(m_path)).astype(np.uint8)[z]
        return ct_slice, m_slice

    def __len__(self):
        return len(self.index_all)

    def __getitem__(self, idx):
        if self.training and self.index_pos and random.random() < self.pos_fraction:
            pid, z = random.choice(self.index_pos)
        else:
            pid, z = self.index_all[idx]

        ct_slice, m_slice = self._get_slice(pid, z)

        img = torch.from_numpy(ct_slice[None, ...])
        mask = torch.from_numpy((m_slice > 0).astype(np.float32)[None, ...])

        return img, mask

# ─────────────────────────────────────────────
# LOSS & EVALUATION METRICS
# ─────────────────────────────────────────────
bce = nn.BCEWithLogitsLoss()

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()

def total_loss(logits, targets):
    return 0.5 * bce(logits, targets) + 0.5 * dice_loss(logits, targets)

@torch.no_grad()
def dice_score(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    pred = (probs > thresh).float()
    num = 2 * (pred * targets).sum(dim=(2, 3)) + eps
    den = (pred + targets).sum(dim=(2, 3)) + eps
    return (num / den).mean().item()

# ─────────────────────────────────────────────
# TRAIN / VAL
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    tot = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = total_loss(logits, y)
        loss.backward()
        optimizer.step()

        tot += loss.item()

    return tot / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    tot_loss, tot_dice = 0.0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        tot_loss += total_loss(logits, y).item()
        tot_dice += dice_score(logits, y)

    return tot_loss / len(loader), tot_dice / len(loader)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    set_seed(42)

    BASE_DIR = "/home/3175552/SecondCNN/dataset_split_1303"
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    VAL_DIR = os.path.join(BASE_DIR, "val")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCHS = 50
    BATCH_SIZE = 64
    LR = 1e-3

    print(f"Avvio baseline U-Net 2D su {DEVICE}", flush=True)
    print(f"Train dir: {TRAIN_DIR}", flush=True)
    print(f"Val dir  : {VAL_DIR}", flush=True)

    # Dataset
    train_ds = Pancreas2DSliceDataset(
        TRAIN_DIR,
        pos_fraction=0.5,
        cache_volumes=True,
        training=True
    )

    val_ds = Pancreas2DSliceDataset(
        VAL_DIR,
        pos_fraction=0.0,
        cache_volumes=True,
        training=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    # Modello
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_dice = -1.0
    best_epoch = -1

    print("\n>>>> INIZIO TRAINING BASELINE <<<<", flush=True)
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        va_loss, va_dice = validate(model, val_loader, DEVICE)

        print(
            f"Ep {epoch:03d} | tr_loss {tr_loss:.4f} | "
            f"va_loss {va_loss:.4f} | va_dice {va_dice:.4f}",
            flush=True
        )

        if va_dice > best_dice:
            best_dice = va_dice
            best_epoch = epoch
            torch.save(model.state_dict(), "best_unet2d_baseline.pt")

    print("\n>>>> TRAINING COMPLETATO <<<<", flush=True)
    print(f"Miglior epoca : {best_epoch}", flush=True)
    print(f"Miglior Dice  : {best_dice:.4f}", flush=True)
    print("Checkpoint salvato in: best_unet2d_baseline.pt", flush=True)