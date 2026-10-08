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
# NOTE: augmentation removed for sanity check
# ──────────────────────────────    ───────────────

# ─────────────────────────────────────────────
# READ PATIENT IDS FROM CT FOLDER
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
# DATASET (con flag shuffle_labels)
# ─────────────────────────────────────────────
class Pancreas2DSliceDataset(Dataset):
    def __init__(self, split_dir, pos_fraction=0.5, cache_volumes=True, training=True, shuffle_labels=False):
        """ shuffle_labels : if true, the mask of each slice is taken from a different patient chosen at random. Use ONLY for sanity check. """

        self.ct_dir = os.path.join(split_dir, "ct")
        self.mask_dir = os.path.join(split_dir, "mask")
        self.pos_fraction = pos_fraction
        self.cache_volumes = cache_volumes
        self.training = training
        self.shuffle_labels = shuffle_labels

        self.patient_ids = get_patient_ids(self.ct_dir)
        self.index_all = []
        self.index_pos = []
        self.volumes_ct = {}
        self.volumes_mask = {}

        mode_str = " [SHUFFLE LABELS - SANITY CHECK]" if shuffle_labels else ""
        print(f"Indicizzazione di {len(self.patient_ids)} pazienti da '{split_dir}'{mode_str}...", flush=True)

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
        """Returns (ct_slice, mask_slice) for a given pid and z."""
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

        # ── SANITY CHECK: replace the mask with that of another patient ──
        if self.shuffle_labels:
            other_pids = [p for p in self.patient_ids if p != pid]
            if other_pids:
                rand_pid = random.choice(other_pids)
                rand_z = random.randint(0, self.volumes_mask[rand_pid].shape[0] - 1)
                _, m_slice = self._get_slice(rand_pid, rand_z)

        img = torch.from_numpy(ct_slice[None, ...])
        mask = torch.from_numpy((m_slice > 0).astype(np.float32)[None, ...])

        # augmentation removed for sanity check: return the original tensors
        return img, mask

# ─────────────────────────────────────────────
# LOSS AND METRICS
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
def dice_score_2d(logits, targets, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    pred = (probs > thresh).float()
    num = 2 * (pred * targets).sum(dim=(2, 3)) + eps
    den = (pred + targets).sum(dim=(2, 3)) + eps
    return (num / den).mean().item()

# ─────────────────────────────────────────────
# 3D DICE PER PATIENT (DEBUG)
# ─────────────────────────────────────────────
@torch.no_grad()
def dice_score_3d_patients_debug(model, split_dir, device, thresh=0.5, eps=1e-6):
    """ Calculates the 3D Dice with the TRUE masks (never shuffled), regardless of how the model was trained.
        Debug version: prints statistics of logits/probs and number of positive pixels predicted per patient.
    """
    model.eval()
    ct_dir = os.path.join(split_dir, "ct")
    mask_dir = os.path.join(split_dir, "mask")
    patient_ids = get_patient_ids(ct_dir)
    scores = []

    for pid in patient_ids:
        ct_path = find_ct_path(ct_dir, pid)
        m_path = find_mask_path(mask_dir, pid)
        if ct_path is None or m_path is None:
            print(f" [SKIP 3D] CT o mask mancante per {pid}", flush=True)
            continue

        ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)  # (S,H,W)
        mask = sitk.GetArrayFromImage(sitk.ReadImage(m_path)).astype(np.uint8)

        # --- WARNING: apply the same preprocessing used in the Dataset, if present ---
        # If in your normal dataset you do clipping/normalization, apply it here too.
        # Example (commented): ct = np.clip(ct, -100, 400); ct = (ct - ct.mean()) / (ct.std() + 1e-8)

        ct_tensor = torch.from_numpy(ct[:, None, ...]).to(device)  # (S,1,H,W)
        logits = model(ct_tensor)  # (S,1,H,W)
        probs = torch.sigmoid(logits)

        min_logit = float(logits.min().cpu())
        max_logit = float(logits.max().cpu())
        mean_logit = float(logits.mean().cpu())
        min_prob = float(probs.min().cpu())
        max_prob = float(probs.max().cpu())
        mean_prob = float(probs.mean().cpu())
        total_pos_pixels = int((probs > thresh).sum().cpu().item())

        print(f"[DEBUG] Paziente {pid}: logit min/max/mean = {min_logit:.4f}/{max_logit:.4f}/{mean_logit:.4f}; "
              f"prob min/max/mean = {min_prob:.6f}/{max_prob:.6f}/{mean_prob:.6f}; "
              f"pred_pos_pixels(@{thresh}) = {total_pos_pixels}", flush=True)

        pred = (probs > thresh).float().cpu().numpy()[:, 0]
        gt = (mask > 0).astype(np.float32)
        num = 2 * (pred * gt).sum() + eps
        den = pred.sum() + gt.sum() + eps
        patient_dice = float(num / den)
        scores.append(patient_dice)
        print(f" Paziente {pid}: Dice = {patient_dice:.8f}", flush=True)

    mean_score = float(np.mean(scores)) if scores else 0.0
    std_score = float(np.std(scores)) if scores else 0.0
    return mean_score, std_score

# ─────────────────────────────────────────────
# TRAINING / VALIDATION
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    tot = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = total_loss(logits, y)
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
        logits = model(x)
        tot_loss += total_loss(logits, y).item()
        tot_dice += dice_score_2d(logits, y)
    return tot_loss / len(loader), tot_dice / len(loader)

# ─────────────────────────────────────────────
# RUN FUNCTION (reusable for both runs)
# ─────────────────────────────────────────────
def run_experiment(train_dir, val_dir, device, epochs, batch_size, lr_max, seed, shuffle_labels=False, checkpoint_path="best_unet2d.pt"):
    set_seed(seed)
    label_tag = "SHUFFLED_LABELS [SANITY CHECK]" if shuffle_labels else "NORMAL"
    print(f"\n{'='*60}", flush=True)
    print(f" RUN: {label_tag}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Dataset
    train_ds = Pancreas2DSliceDataset(
        train_dir, pos_fraction=0.5, cache_volumes=True, training=True, shuffle_labels=shuffle_labels
    )
    val_ds = Pancreas2DSliceDataset(
        val_dir, pos_fraction=0.0, cache_volumes=True, training=False, shuffle_labels=False
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # Model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr_max / 10)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr_max,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=1e4
    )

    best_dice_2d = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        va_loss, va_dice = validate(model, val_loader, device)
        print(
            f"Ep {epoch:03d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} | "
            f"va_dice(2D) {va_dice:.4f} | lr {scheduler.get_last_lr()[0]:.6f}",
            flush=True
        )
        if va_dice > best_dice_2d:
            best_dice_2d = va_dice
            best_epoch = epoch
            # save exactly in the requested path
            torch.save(model.state_dict(), checkpoint_path)

    # 3D Dice with true masks: load the best checkpoint safely and debug
    print(f"\n>>>> FINAL 3D DICE (bestckpoint: ep {best_epoch}) <<<<", flush=True)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
    else:
        print(f" [WARN] checkpoint non trovato: {checkpoint_path}. Uso modello corrente.", flush=True)

    dice_mean, dice_std = dice_score_3d_patients_debug(model, val_dir, device)
    return {
        "label_mode": label_tag,
        "best_epoch": best_epoch,
        "dice_2d_val": best_dice_2d,
        "dice_3d_mean": dice_mean,
        "dice_3d_std": dice_std,
    }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR = "/home/3175552/SecondCNN/dataset_split_1303"  
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    VAL_DIR = os.path.join(BASE_DIR, "val")

    EPOCHS = 30
    BATCH_SIZE = 64
    LR_MAX = 1e-3
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Avvio Sanity Check su {DEVICE}", flush=True)

    # ── RUN 1: NORMAL TRAINING ──────────────────────────────────────────
    results_normal = run_experiment(
        TRAIN_DIR, VAL_DIR, DEVICE, epochs=EPOCHS, batch_size=BATCH_SIZE, lr_max=LR_MAX, seed=SEED,
        shuffle_labels=False, checkpoint_path="best_unet2d_normal.pt"
    )

    # ── RUN 2: TRAINING WITH SHUFFLED MASKS ───────────────────────────────
    results_sanity = run_experiment(
        TRAIN_DIR, VAL_DIR, DEVICE, epochs=min(EPOCHS, 10),  # 10 epoch per diagnosticare
        batch_size=BATCH_SIZE, lr_max=LR_MAX, seed=SEED, shuffle_labels=True,
        checkpoint_path="best_unet2d_sanity.pt"
    )

    # ── COMPARATIVE REPORT ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" RISULTATI COMPARATIVI SANITY CHECK")
    print("=" * 60)
    for r in [results_normal, results_sanity]:
        print(f"\n Modalità : {r['label_mode']}")
        print(f" Best epoch : {r['best_epoch']}")
        print(f" Dice 2D (val) : {r['dice_2d_val']:.4f}")
        print(f" Dice 3D (val) : {r['dice_3d_mean']:.4f} ± {r['dice_3d_std']:.4f}")
        print()

    # ── AUTOMATIC INTERPRETATION ────────────────────────────────────────────
    print(" INTERPRETATION:")
    gap = results_normal["dice_3d_mean"] - results_sanity["dice_3d_mean"]
    print(f" Gap Dice 3D (normale - shuffled) = {gap:.4f}")
    if results_sanity["dice_3d_mean"] > 0.4:
        print("\n ⚠️ WARNING: the model with shuffled labels reaches Dice > 0.4.")
        print(" This is a strong signal of data leakage or bug in the pipeline.")
        print(" Check: data split, volume cache, shared indexes.")
    elif results_sanity["dice_3d_mean"] > 0.2:
        print("\n ⚠️ MODERATE WARNING: Dice with shuffled labels between 0.2 and 0.4.")
        print(" There might be partial leakage or the model exploits class bias.")
        print(" Investigate the foreground/background distribution in the dataset.")
    else:
        print("\n ✅ Sanity check PASSED: shuffled labels produce low Dice.")
        print(" The pipeline seems correct. The high performance in the normal run")
        print(" are probably genuine.")
    print("=" * 60)