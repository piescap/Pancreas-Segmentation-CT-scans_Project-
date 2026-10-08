import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk

# Set the random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Read the patients ID
def get_patient_ids(ct_folder):
    """
    From the structure of the dataset, extract the patient IDs. 
    Handles both 'pancreas_{pid}.nii.gz' and '{pid}.nii.gz' naming conventions.
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

def find_ct_path(ct_folder, pid):
    """ 
    Given a patient ID, find the corresponding CT file path.
    """
    for candidate in [f"pancreas_{pid}.nii.gz", f"{pid}.nii.gz"]:
        p = os.path.join(ct_folder, candidate)
        if os.path.exists(p):
            return p
    return None

def find_mask_path(mask_folder, pid):
    """
    Given a patient ID, find the corresponding mask file path.
    """
    for candidate in [f"label{pid}.nii.gz", f"pancreas_{pid}.nii.gz", f"{pid}.nii.gz"]:
        p = os.path.join(mask_folder, candidate)
        if os.path.exists(p):
            return p
    return None



# Load the dataset and prepare the list of (patient_id, slice_index) for training/validation

class Pancreas2DSliceDataset(Dataset):
    """
    This is the class defined to load the dataset. It reads the CT and mask volumes, 
    creates an index of all slices, and allows for balanced sampling of positive slices during training.

    The idea is: since we have more slice without pancreas rather than with pancreas, we want to sample more frequently the positive slices during training.
        - pos_fraction: the fraction of positive slices to sample during training.
    """
    def __init__(self, split_dir,
                 pos_fraction=0.5, cache_volumes=True, training=True):
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
                print(f"  > Processed {i+1}/{len(self.patient_ids)} patients...", flush=True)

        print(f"  Total slices: {len(self.index_all)} | Positive slices: {len(self.index_pos)}", flush=True)

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

        return img, mask

# ─────────────────────────────────────────────
#  The Decoder Block That Will be used Later. 
# ─────────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


#  MODEL: ResNet34 + Progressive Decoder 

class ResNet34Baseline(nn.Module):
    """ 
    The Resnet34Baseline is a simple encoder-decoder architecture that uses a ResNet34 as the encoder 
    and a progressive decoder with 5 blocks.

    ResNet34 is considered a "heavy" encoder, and the decoder is "progressive" because it gradually reduces the number of channels from 512 to 16.
    """
    def __init__(self, pretrained=False):
        super().__init__()
        # ResNet34 expects 3 input channels, but our CT slices have only 1 channel.
        self.channel_adapt = nn.Conv2d(1, 3, kernel_size=1, bias=False) 

        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128,  64)
        self.dec4 = DecoderBlock( 64,  32)
        self.dec5 = DecoderBlock( 32,  16)

        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = self.channel_adapt(x)
        x = self.encoder(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.head(x)
        return x




#  LOSS and METRIC



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

#  Calculate the 3D Dice score at patient level. This is the main metric for evaluation. Not for training. 
#  The 3D Dice is calculated by aggregating all the slices of a patient together, and then calculating the Dice score on the whole volume.


@torch.no_grad()
def dice_score_3d_patients(model, split_dir, device, thresh=0.5, eps=1e-6):
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
#  TRAINING / VALIDATION Loops
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    tot = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss   = total_loss(logits, y)
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
        logits   = model(x)
        tot_loss += total_loss(logits, y).item()
        tot_dice += dice_score_2d(logits, y)
    return tot_loss / len(loader), tot_dice / len(loader)


#  MAIN -- effectively running the script. 

if __name__ == "__main__":

    BASE_DIR   = "/home/3175552/SecondCNN/dataset_split_1303"
    TRAIN_DIR  = os.path.join(BASE_DIR, "train")
    VAL_DIR    = os.path.join(BASE_DIR, "val")

    EPOCHS     = 50
    BATCH_SIZE = 64
    LR         = 1e-3
    SEED       = 42

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)

    print(f"Avvio Training BASELINE (ResNet34 + decoder progressivo) su {DEVICE}", flush=True)
    print(f"Modello      : ResNet34 encoder + decoder progressivo ", flush=True)
    print(f"Loss         : 0.5 * BCE  +  0.5 * Dice", flush=True)
    print(f"LR           : {LR} (fisso)", flush=True)



    # Import dataset and create dataloaders
    
    print("\n>>>> PREPARAZIONE DATASET <<<<", flush=True)
    train_ds = Pancreas2DSliceDataset(
        TRAIN_DIR, pos_fraction=0.5, cache_volumes=True, training=True
    )

    val_ds = Pancreas2DSliceDataset(
        VAL_DIR, pos_fraction=0.0, cache_volumes=True, training=False
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=1, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=1, pin_memory=True)

    # ── Model ───────────────────────────────
    model = ResNet34Baseline(pretrained=False).to(DEVICE)

    # Check the total number of trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable Parameters: {n_params:,}", flush=True)

    # Fixed Learning Rate, no scheduler. This is to observe the pure effect of the architecture without confounding factors.
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_dice_2d = -1.0
    best_epoch   = -1

    # ── Training loop ─────────────────────────
    print("\n>>>> INIZIO TRAINING <<<<", flush=True)
    for epoch in range(1, EPOCHS + 1):
        tr_loss          = train_one_epoch(model, train_loader, optimizer, DEVICE)
        va_loss, va_dice = validate(model, val_loader, DEVICE)

        print(
            f"Ep {epoch:03d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} | "
            f"va_dice(2D) {va_dice:.4f} | lr {LR:.6f}",
            flush=True
        )

        if va_dice > best_dice_2d:
            best_dice_2d = va_dice
            best_epoch   = epoch
            torch.save(model.state_dict(), "best_resnet34_progressive_decoder.pt")

    # ── Dice 3D finale ────────────────────────
    print(f"\n>>>> DICE 3D FINALE (miglior checkpoint: ep {best_epoch}) <<<<", flush=True)
    model.load_state_dict(torch.load("best_resnet34_progressive_decoder.pt"))
    dice_mean, dice_std = dice_score_3d_patients(model, VAL_DIR, DEVICE)

    print("\n" + "=" * 50)
    print(f"  Modello       : ResNet34 + decoder progressivo (no skip connections)")
    print(f"  Augmentation  : Disabilitata")
    print(f"  LR            : {LR} (fisso)")
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Dice 2D (val) : {best_dice_2d:.4f}")
    print(f"  Dice 3D (val) : {dice_mean:.4f} ± {dice_std:.4f}")
    print("=" * 50)