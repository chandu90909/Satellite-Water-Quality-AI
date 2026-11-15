from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "unet"
MODEL_DIR = PROJECT_ROOT / "models" / "unet"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4

NUM_CHANNELS = 5
IGNORE_VALUE = 255

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("U-NET TRAINING")
print("=" * 70)

print(f"Device       : {DEVICE}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")

print("=" * 70)


# ============================================================
# DATASET
# ============================================================

class WaterDataset(Dataset):

    def __init__(self, split):

        self.split = split

        self.image_dir = DATASET_DIR / split / "images"
        self.mask_dir = DATASET_DIR / split / "masks"
        self.valid_dir = DATASET_DIR / split / "valid"

        self.files = sorted(
            self.image_dir.glob("*.npy")
        )

        if len(self.files) == 0:

            raise RuntimeError(
                f"No image patches found in:\n"
                f"{self.image_dir}"
            )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        image_path = self.files[idx]

        mask_path = (
            self.mask_dir /
            image_path.name
        )

        valid_path = (
            self.valid_dir /
            image_path.name
        )

        # ====================================================
        # LOAD IMAGE
        # ====================================================

        image = np.load(
            image_path
        ).astype(np.float32)

        # Dataset format:
        #
        # Channels x Height x Width
        #
        # Expected:
        #
        # (5, 256, 256)

        if image.shape[0] != NUM_CHANNELS:

            raise ValueError(
                f"Unexpected image shape: "
                f"{image.shape} "
                f"in {image_path.name}. "
                f"Expected first dimension = "
                f"{NUM_CHANNELS}."
            )

        # ====================================================
        # HANDLE NaN / INF
        # ====================================================

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        # ====================================================
        # CLIP INPUT
        # ====================================================

        image = np.clip(
            image,
            0.0,
            1.0
        )

        # Final safety check

        if not np.isfinite(image).all():

            raise ValueError(
                f"Invalid values remain in "
                f"{image_path.name}"
            )

        # ====================================================
        # LOAD MASK
        # ====================================================

        mask = np.load(
            mask_path
        ).astype(np.float32)

        # ====================================================
        # LOAD VALID MASK
        # ====================================================

        valid = np.load(
            valid_path
        ).astype(np.float32)

        # ====================================================
        # IGNORE VALUE
        # ====================================================

        # Original mask:
        #
        # 0   = non-water
        # 1   = water
        # 255 = ignore
        #
        # Temporarily convert 255 to 0.
        # The valid mask ensures ignored pixels
        # do not contribute to the loss.

        target = mask.copy()

        target[
            target == IGNORE_VALUE
        ] = 0

        # ====================================================
        # ADD CHANNEL DIMENSION
        # ====================================================

        target = np.expand_dims(
            target,
            axis=0
        )

        valid = np.expand_dims(
            valid,
            axis=0
        )

        # ====================================================
        # CONVERT TO TORCH
        # ====================================================

        image = torch.from_numpy(
            image
        ).float()

        target = torch.from_numpy(
            target
        ).float()

        valid = torch.from_numpy(
            valid
        ).float()

        return image, target, valid


# ============================================================
# DOUBLE CONVOLUTION BLOCK
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# U-NET MODEL
# ============================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        self.enc1 = DoubleConv(
            5,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

        # ----------------------------------------------------
        # BOTTLENECK
        # ----------------------------------------------------

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        self.output = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):

        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # ====================================================
        # BOTTLENECK
        # ====================================================

        b = self.bottleneck(
            self.pool(e4)
        )

        # ====================================================
        # DECODER
        # ====================================================

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # ====================================================
        # OUTPUT
        # ====================================================

        return self.output(d1)


# ============================================================
# MASKED BCE LOSS
# ============================================================

def masked_bce_loss(
    prediction,
    target,
    valid
):

    loss = nn.functional.binary_cross_entropy_with_logits(
        prediction,
        target,
        reduction="none"
    )

    # Ignore invalid pixels

    loss = loss * valid

    valid_pixels = valid.sum()

    if valid_pixels <= 0:

        return loss.mean()

    return (
        loss.sum() /
        valid_pixels.clamp(min=1)
    )


# ============================================================
# MASKED DICE LOSS
# ============================================================

def masked_dice_loss(
    prediction,
    target,
    valid
):

    probability = torch.sigmoid(
        prediction
    )

    # Ignore invalid pixels

    probability = probability * valid
    target = target * valid

    intersection = (
        probability * target
    ).sum()

    denominator = (
        probability.sum()
        + target.sum()
    )

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        denominator + 1e-6
    )

    return 1.0 - dice


# ============================================================
# COMBINED LOSS
# ============================================================

def combined_loss(
    prediction,
    target,
    valid
):

    bce = masked_bce_loss(
        prediction,
        target,
        valid
    )

    dice = masked_dice_loss(
        prediction,
        target,
        valid
    )

    return bce + dice


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    target,
    valid
):

    probability = torch.sigmoid(
        prediction
    )

    predicted = (
        probability >= 0.5
    ).float()

    # Ignore invalid pixels

    predicted = predicted * valid
    target = target * valid

    intersection = (
        predicted * target
    ).sum()

    union = (
        predicted
        + target
        - predicted * target
    ).sum()

    # Dice

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        predicted.sum()
        + target.sum()
        + 1e-6
    )

    # IoU

    iou = (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )

    return (
        dice.item(),
        iou.item()
    )


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = WaterDataset(
    "train"
)

val_dataset = WaterDataset(
    "val"
)

test_dataset = WaterDataset(
    "test"
)

print(
    f"Training samples  : "
    f"{len(train_dataset)}"
)

print(
    f"Validation samples: "
    f"{len(val_dataset)}"
)

print(
    f"Test samples      : "
    f"{len(test_dataset)}"
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CREATE MODEL
# ============================================================

model = UNet().to(
    DEVICE
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# MODEL PATH
# ============================================================

best_model_path = (
    MODEL_DIR /
    "unet_best.pth"
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

best_val_loss = float("inf")

patience = 8

patience_counter = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    train_loss = 0.0

    for images, masks, valid in train_loader:

        images = images.to(
            DEVICE
        )

        masks = masks.to(
            DEVICE
        )

        valid = valid.to(
            DEVICE
        )

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if not torch.isfinite(
            images
        ).all():

            raise RuntimeError(
                "Training batch contains "
                "NaN or Inf values."
            )

        # ----------------------------------------------------
        # CLEAR GRADIENTS
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        predictions = model(
            images
        )

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = combined_loss(
            predictions,
            masks,
            valid
        )

        # ----------------------------------------------------
        # LOSS SAFETY CHECK
        # ----------------------------------------------------

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Loss became invalid "
                f"at epoch {epoch}."
            )

        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # GRADIENT CLIPPING
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(
        train_loader
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_dice = 0.0

    val_iou = 0.0


    with torch.no_grad():

        for images, masks, valid in val_loader:

            images = images.to(
                DEVICE
            )

            masks = masks.to(
                DEVICE
            )

            valid = valid.to(
                DEVICE
            )

            predictions = model(
                images
            )

            loss = combined_loss(
                predictions,
                masks,
                valid
            )

            dice, iou = calculate_metrics(
                predictions,
                masks,
                valid
            )

            val_loss += loss.item()

            val_dice += dice

            val_iou += iou


    val_loss /= len(
        val_loader
    )

    val_dice /= len(
        val_loader
    )

    val_iou /= len(
        val_loader
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Dice: {val_dice:.4f} | "
        f"IoU: {val_iou:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "val_loss":
                    val_loss,

                "val_dice":
                    val_dice,

                "val_iou":
                    val_iou
            },
            best_model_path
        )

        print(
            f"  -> Best model saved: "
            f"{best_model_path}"
        )

        patience_counter = 0

    else:

        patience_counter += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if patience_counter >= patience:

        print()
        print(
            "Early stopping triggered."
        )

        break


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("U-NET TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best model: {best_model_path}"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.4f}"
)

print("=" * 70)