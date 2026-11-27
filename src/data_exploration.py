
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "classification"
train_dir = DATA_DIR / "Training"
test_dir = DATA_DIR / "Testing"

IMG_SIZE = 224
IMG_EXTS = {".jpg", ".jpeg", ".png"}

EXPECTED_CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# ImageNet normalization for pretrained ResNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Reproducibility
SEED = 42
random.seed(SEED)

# Output directory
OUT_DIR = ROOT / "outputs" / "preprocessing"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> list[Path]:
    """Return a sorted list of supported image files."""
    return sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    )


# ==========================================
# STEP 1: Validate dataset structure
# ==========================================

for split_dir in (train_dir, test_dir):

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {split_dir}"
        )

    actual_classes = {
        c.name for c in split_dir.iterdir()
        if c.is_dir()
    }

    missing = set(EXPECTED_CLASSES) - actual_classes
    unexpected = actual_classes - set(EXPECTED_CLASSES)

    if missing:
        raise ValueError(
            f"Missing classes in {split_dir}: {missing}"
        )

    if unexpected:
        raise ValueError(
            f"Unexpected directories in {split_dir}: {unexpected}"
        )

    for class_name in EXPECTED_CLASSES:
        if not list_images(split_dir / class_name):
            raise ValueError(
                f"Empty class: {split_dir / class_name}"
            )

    print(f"{split_dir.name}: All four classes are present.")


# ==========================================
# STEP 2: Count images per class
# ==========================================

def count_per_class(split_dir: Path) -> dict:
    """Count the number of images in each class."""
    return {
        c: len(list_images(split_dir / c))
        for c in EXPECTED_CLASSES
    }


train_counts = count_per_class(train_dir)
test_counts = count_per_class(test_dir)

for name, counts in (
    ("Training", train_counts),
    ("Testing", test_counts)
):

    total = sum(counts.values())

    print(f"\n{name}: {total} images")

    for c, n in counts.items():
        print(
            f"  {c:<12} {n:>5} "
            f"({100 * n / total:.1f}%)"
        )

    imbalance_ratio = (
        max(counts.values()) / min(counts.values())
    )

    print(f"Imbalance ratio: {imbalance_ratio:.2f}")


# ==========================================
# STEP 3: Visualize class distribution
# ==========================================

classes = EXPECTED_CLASSES
x = np.arange(len(classes))
w = 0.4

fig, ax = plt.subplots(figsize=(9, 5))

b1 = ax.bar(
    x - w / 2,
    [train_counts[c] for c in classes],
    w,
    label="Training"
)

b2 = ax.bar(
    x + w / 2,
    [test_counts[c] for c in classes],
    w,
    label="Testing"
)

ax.bar_label(b1)
ax.bar_label(b2)

ax.set_xticks(x)
ax.set_xticklabels(classes)

ax.set_xlabel("Tumor Classes")
ax.set_ylabel("Number of Images")
ax.set_title("Class Distribution")
ax.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "class_distribution.png", dpi=150)
plt.show()
plt.close(fig)


# ==========================================
# STEP 4: Define preprocessing transformations
# ==========================================

class PadToSquare:
    """Pad an image to a square while preserving its aspect ratio."""

    def __call__(self, img: Image.Image) -> Image.Image:

        img = img.convert("RGB")

        w, h = img.size
        side = max(w, h)

        canvas = Image.new("RGB", (side, side), (0, 0, 0))

        canvas.paste(
            img,
            ((side - w) // 2, (side - h) // 2)
        )

        return canvas


# Direct resizing
resize_simple = transforms.Resize(
    (IMG_SIZE, IMG_SIZE)
)

# Padding followed by resizing
resize_pad = transforms.Compose([
    PadToSquare(),
    transforms.Resize((IMG_SIZE, IMG_SIZE))
])

# Complete preprocessing pipeline
preprocess = transforms.Compose([
    PadToSquare(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


# ==========================================
# STEP 5: Visualize preprocessing results
# ==========================================

def load_image(path: Path) -> Image.Image:
    """Load an image and safely close the original file."""

    with Image.open(path) as img:
        return img.copy()


samples = {
    c: load_image(
        random.choice(list_images(train_dir / c))
    )
    for c in classes
}

fig, axes = plt.subplots(
    3,
    len(classes),
    figsize=(4 * len(classes), 11),
    squeeze=False
)

for j, c in enumerate(classes):

    img = samples[c]

    simple = resize_simple(img)
    padded = resize_pad(img)

    # Original image
    axes[0, j].imshow(img, cmap="gray")
    axes[0, j].set_title(
        f"{c}\nOriginal {img.size} ({img.mode})"
    )

    # Direct resizing
    axes[1, j].imshow(simple, cmap="gray")
    axes[1, j].set_title("Direct Resize")

    # Padding and resizing
    axes[2, j].imshow(padded, cmap="gray")
    axes[2, j].set_title("Padding + Resize")

    for i in range(3):
        axes[i, j].axis("off")

plt.tight_layout()
plt.savefig(OUT_DIR / "before_after.png", dpi=120)
plt.show()
plt.close(fig)


# ==========================================
# STEP 6: Validate the final tensor
# ==========================================

img = samples[classes[0]]
tensor = preprocess(img)

print("\nPreprocessing Results")

print("Original image:", img.size, img.mode)
print("Tensor shape:", tuple(tensor.shape))
print("Tensor dtype:", tensor.dtype)

print(
    f"Pixel values: "
    f"min={tensor.min():.2f}, "
    f"max={tensor.max():.2f}, "
    f"mean={tensor.mean():.2f}"
)

assert tensor.shape == (3, IMG_SIZE, IMG_SIZE), (
    "Unexpected tensor shape"
)


# ==========================================
# STEP 7: Denormalize for visualization
# ==========================================

mean = np.array(MEAN).reshape(3, 1, 1)
std = np.array(STD).reshape(3, 1, 1)

restored = np.clip(
    tensor.numpy() * std + mean,
    0,
    1
).transpose(1, 2, 0)

fig = plt.figure(figsize=(4, 4))

plt.imshow(restored)
plt.title("Image After Complete Preprocessing")
plt.axis("off")

plt.tight_layout()
plt.savefig(OUT_DIR / "preprocessed_image.png", dpi=150)
plt.show()
plt.close(fig)
#1. Pourquoi Data Augmentation ?
#Notre dataset est déjà équilibré. Donc notre objectif n'est pas d'équilibrer les classes, 
# mais de permettre au modèle de mieux généraliser sur de nouvelles images IRM.
