# training anche con data augmentation

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import segmentation_models_pytorch as smp

# ─────────────────────────────────────────────
#  REPRODUCIBILITY 
# ─────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ─────────────────────────────────────────────
#  AUGMENTATION
# ─────────────────────────────────────────────
def augment(img: torch.Tensor, mask: torch.Tensor):
    if random.random() > 0.5:
        img  = torch.flip(img,  dims=[-1])
        mask = torch.flip(mask, dims=[-1])
    if random.random() > 0.5:
        img  = torch.flip(img,  dims=[-2])
        mask = torch.flip(mask, dims=[-2])
    if random.random() > 0.3:
        angle = random.uniform(-15, 15)
        img  = TF.rotate(img,  angle)
        mask = TF.rotate(mask, angle)
    if random.random() > 0.5:
        factor = random.uniform(0.85, 1.15)
        img = img * factor
    return img, mask

# ─────────────────────────────────────────────
#  READ PATIENT IDS & PATHS
# ─────────────────────────────────────────────
def get_patient_ids(ct_folder):
    """
    Legge tutti i file .nii.gz nella cartella ct e restituisce
    la lista degli ID. Supporta due formati:
      - pancreas_XXXX.nii.gz  →  pid = 'XXXX'
      - XXXX.nii.gz           →  pid = 'XXXX'
    """
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

# ─────────────────────────────────────────────
# FIND CT AND MASK FILES GIVEN A PATIENT ID
# ─────────────────────────────────────────────
def find_ct_path(ct_folder, pid):
    """Cerca pancreas_PID.nii.gz oppure PID.nii.gz nella cartella ct."""
    for candidate in [f"pancreas_{pid}.nii.gz", f"{pid}.nii.gz"]:
        p = os.path.join(ct_folder, candidate)
        if os.path.exists(p):
            return p
    return None

def find_mask_path(mask_folder, pid):
    """Cerca label PID.nii.gz, pancreas_PID.nii.gz o PID.nii.gz nella cartella mask."""
    for candidate in [f"label{pid}.nii.gz", f"pancreas_{pid}.nii.gz", f"{pid}.nii.gz"]:
        p = os.path.join(mask_folder, candidate)
        if os.path.exists(p):
            return p
    return None

# ─────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────
class Pancreas2DSliceDataset(Dataset):
    def __init__(self, split_dir,
                 pos_fraction=0.5, cache_volumes=True, training=True):
        """
        split_dir : cartella del split, es. 'dataset_split_1303/train'
                    Si aspetta al suo interno:
                      split_dir/ct/    → file CT  (.nii.gz)
                      split_dir/mask/  → file mask (.nii.gz)
        """
        self.ct_dir        = os.path.join(split_dir, "ct")
        self.mask_dir      = os.path.join(split_dir, "mask")
        self.pos_fraction  = pos_fraction
        self.cache_volumes = cache_volumes
        self.training      = training

        self.patient_ids  = get_patient_ids(self.ct_dir)
        self.index_all    = []
        self.index_pos    = []
        self.volumes_ct   = {}
        self.volumes_mask = {}

        print(f"Indicizzazione di {len(self.patient_ids)} pazienti da '{split_dir}'...", flush=True)

        for i, pid in enumerate(self.patient_ids):
            ct_path = find_ct_path(self.ct_dir, pid)
            m_path  = find_mask_path(self.mask_dir, pid)

            if ct_path is None:
                print(f"  [SKIP] CT mancante per {pid}", flush=True)
                continue
            if m_path is None:
                print(f"  [SKIP] maschera mancante per {pid}", flush=True)
                continue

            ct_array = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
            m_array  = sitk.GetArrayFromImage(sitk.ReadImage(m_path)).astype(np.uint8)

            if self.cache_volumes:
                self.volumes_ct[pid]   = ct_array
                self.volumes_mask[pid] = m_array

            for z in range(ct_array.shape[0]):
                self.index_all.append((pid, z))
                if m_array[z].max() > 0:
                    self.index_pos.append((pid, z))

            if (i + 1) % 10 == 0:
                print(f"  > Processati {i+1}/{len(self.patient_ids)} pazienti...", flush=True)

        print(f"  Totale slice: {len(self.index_all)} | Slice positive: {len(self.index_pos)}", flush=True)

    def __len__(self):
        return len(self.index_all)

    def __getitem__(self, idx):
        if self.training and self.index_pos and random.random() < self.pos_fraction:
            pid, z = random.choice(self.index_pos)
        else:
            pid, z = self.index_all[idx]

        if self.cache_volumes:
            ct_slice = self.volumes_ct[pid][z]
            m_slice  = self.volumes_mask[pid][z]
        else:
            ct_path = find_ct_path(self.ct_dir, pid)
            m_path  = find_mask_path(self.mask_dir, pid)
            ct_slice = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)[z]
            m_slice  = sitk.GetArrayFromImage(sitk.ReadImage(m_path)).astype(np.uint8)[z]

        img  = torch.from_numpy(ct_slice[None, ...])
        mask = torch.from_numpy((m_slice > 0).astype(np.float32)[None, ...])

        if self.training:
            img, mask = augment(img, mask)

        return img, mask

# ─────────────────────────────────────────────
#  LOSS E METRICS
# ─────────────────────────────────────────────
bce = nn.BCEWithLogitsLoss()

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num   = 2 * (probs * targets).sum(dim=(2, 3)) + eps
    den   = (probs + targets).sum(dim=(2, 3)) + eps
    return 1 - (num / den).mean()

def total_loss(logits, targets):
    return 0.5 * bce(logits, targets) + 0.5 * dice_loss(logits, targets)

@torch.no_grad()
def dice_score_2d(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    pred  = (probs > thresh).float()
    num   = 2 * (pred * targets).sum(dim=(2, 3)) + eps
    den   = (pred + targets).sum(dim=(2, 3)) + eps
    return (num / den).mean().item()

# ─────────────────────────────────────────────
#  DICE 3D PER PATIENT
# ─────────────────────────────────────────────
@torch.no_grad()
def dice_score_3d_patients(model, split_dir, device, thresh=0.5, eps=1e-6):
    """
    Legge i pazienti da split_dir/ct e split_dir/mask
    e calcola il Dice 3D per ciascuno.
    """
    model.eval()
    ct_dir      = os.path.join(split_dir, "ct")
    mask_dir    = os.path.join(split_dir, "mask")
    patient_ids = get_patient_ids(ct_dir)
    scores = []

    for pid in patient_ids:
        ct_path = find_ct_path(ct_dir, pid)
        m_path  = find_mask_path(mask_dir, pid)

        if ct_path is None or m_path is None:
            continue

        ct   = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(m_path)).astype(np.uint8)

        ct_tensor = torch.from_numpy(ct[:, None, ...]).to(device)
        logits    = model(ct_tensor)
        pred      = (torch.sigmoid(logits) > thresh).float().cpu().numpy()[:, 0]

        gt   = (mask > 0).astype(np.float32)
        num  = 2 * (pred * gt).sum() + eps
        den  = pred.sum() + gt.sum() + eps
        patient_dice = float(num / den)
        scores.append(patient_dice)
        print(f"  Paziente {pid}: Dice = {patient_dice:.4f}", flush=True)

    return float(np.mean(scores)), float(np.std(scores))

# ─────────────────────────────────────────────
#  TRAINING / VALIDATION
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    tot = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = total_loss(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        tot += loss.item()
    return tot / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    tot_loss, tot_dice = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits   = model(x)
        tot_loss += total_loss(logits, y).item()
        tot_dice += dice_score_2d(logits, y)
    return tot_loss / len(loader), tot_dice / len(loader)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── Percorso base del dataset ──────────────
    # Struttura attesa su disco:
    #   BASE_DIR/train/ct/    → file CT
    #   BASE_DIR/train/mask/  → file mask
    #   BASE_DIR/val/ct/
    #   BASE_DIR/val/mask/
    #   BASE_DIR/test/ct/
    #   BASE_DIR/test/mask/

    BASE_DIR   = "/home/3175552/SecondCNN/dataset_split_1303"   # <-- modifica questo percorso
    TRAIN_DIR  = os.path.join(BASE_DIR, "train")
    VAL_DIR    = os.path.join(BASE_DIR, "val")
    # TEST_DIR = os.path.join(BASE_DIR, "test")  # usato solo per valutazione finale

    # ── Iperparametri ─────────────────────────
    EPOCHS     = 50
    BATCH_SIZE = 64
    LR_MAX     = 1e-3
    SEED       = 42

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    print(f"Avvio Training su {DEVICE}", flush=True)

    # ── Dataset e DataLoader ──────────────────
    print("\n>>>> PREPARAZIONE DATASET <<<<", flush=True)

    train_ds = Pancreas2DSliceDataset(
        TRAIN_DIR,
        pos_fraction=0.5, cache_volumes=True, training=True
    )
    val_ds = Pancreas2DSliceDataset(
        VAL_DIR,
        pos_fraction=0.0, cache_volumes=True, training=False
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=1, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=1, pin_memory=True)

    # ── Model ───────────────────────────────
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX / 10)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR_MAX,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=1e4
    )

    best_dice_2d = -1.0
    best_epoch   = -1

    # ── Training loop ─────────────────────────
    print("\n>>>> INIZIO TRAINING <<<<", flush=True)
    for epoch in range(1, EPOCHS + 1):
        tr_loss          = train_one_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        va_loss, va_dice = validate(model, val_loader, DEVICE)

        print(
            f"Ep {epoch:03d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} | "
            f"va_dice(2D) {va_dice:.4f} | lr {scheduler.get_last_lr()[0]:.6f}",
            flush=True
        )

        if va_dice > best_dice_2d:
            best_dice_2d = va_dice
            best_epoch   = epoch
            torch.save(model.state_dict(), "best_unet2d.pt")

    # ── Dice 3D final on validation set ─────
    print(f"\n>>>> DICE 3D FINALE (miglior checkpoint: ep {best_epoch}) <<<<", flush=True)
    model.load_state_dict(torch.load("best_unet2d.pt"))
    dice_mean, dice_std = dice_score_3d_patients(model, VAL_DIR, DEVICE)

    print("\n" + "=" * 50)
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Dice 2D (val) : {best_dice_2d:.4f}")
    print(f"  Dice 3D (val) : {dice_mean:.4f} ± {dice_std:.4f}")
    print("=" * 50)