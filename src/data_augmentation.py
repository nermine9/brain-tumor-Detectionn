import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from PIL import Image
from torchvision import transforms

from data_transform import (
    IMG_SIZE,
    MEAN,
    STD,
    PadToSquare,
    eval_transform
)


# ==========================================
# 1. DATA AUGMENTATION PIPELINE
# ==========================================

train_transform = transforms.Compose([

    # Preserve the original aspect ratio
    PadToSquare(),

    # Resize the image to 224 x 224
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Convert the image to three grayscale channels
    transforms.Grayscale(num_output_channels=3),

    # Apply random rotation, translation and zoom
    transforms.RandomAffine(
        degrees=10,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05),
        interpolation=transforms.InterpolationMode.BILINEAR
    ),

    # Apply random brightness and contrast
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),

    # Convert the image to a PyTorch tensor
    transforms.ToTensor(),

    # Apply ImageNet normalization
    transforms.Normalize(mean=MEAN, std=STD)
])


# ==========================================
# 2. VISUALIZE DATA AUGMENTATION
# ==========================================

def visualize_augmentation():

    # Define the project and dataset paths
    ROOT = Path(__file__).resolve().parent.parent
    train_dir = ROOT / "data" / "classification" / "Training"

    # Define the output directory
    OUT_DIR = ROOT / "outputs" / "augmentation"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find images in the glioma class
    image_paths = [
        path for path in (train_dir / "glioma").iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    # Check if images are available
    if not image_paths:
        raise ValueError("No glioma images found.")

    # Load one original image
    with Image.open(image_paths[0]) as img:
        image = img.convert("RGB")

    # Prepare normalization parameters
    mean = np.array(MEAN).reshape(3, 1, 1)
    std = np.array(STD).reshape(3, 1, 1)

    # Create a visualization grid
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    # Display the original preprocessed image
    original_tensor = eval_transform(image)

    original = np.clip(
        original_tensor.numpy() * std + mean, 0, 1
    ).transpose(1, 2, 0)

    axes.flat[0].imshow(original)
    axes.flat[0].set_title("Without Augmentation")
    axes.flat[0].axis("off")

    # Generate seven augmented versions
    for i, ax in enumerate(axes.flat[1:]):

        # Apply random transformations
        tensor = train_transform(image)

        # Reverse normalization for visualization
        restored = tensor.numpy() * std + mean

        # Keep pixel values between 0 and 1
        restored = np.clip(restored, 0, 1)

        # Convert CHW to HWC for Matplotlib
        restored = restored.transpose(1, 2, 0)

        # Display the augmented image
        ax.imshow(restored)
        ax.set_title(f"Augmentation {i + 1}")
        ax.axis("off")

    # Add the figure title
    plt.suptitle("Brain MRI - Data Augmentation")
    plt.tight_layout()

    # Save the visualization
    plt.savefig(
        OUT_DIR / "data_augmentation.png",
        dpi=150
    )

    # Display the results
    plt.show()
    plt.close(fig)
from torchvision.transforms import functional as F


def visualize_brightness(image):

    # Define brightness factors
    factors = [0.5, 0.9, 1.0, 1.1, 1.5]

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))

    for ax, factor in zip(axes, factors):

        # Modify brightness only
        adjusted = F.adjust_brightness(image, factor)

        # Display the result
        ax.imshow(adjusted, cmap="gray")
        ax.set_title(f"Brightness: {factor * 100:.0f}%")
        ax.axis("off")

    plt.suptitle("Brightness Comparison")
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. RUN THE VISUALIZATION
# ==========================================

if __name__ == "__main__":
    visualize_augmentation()
