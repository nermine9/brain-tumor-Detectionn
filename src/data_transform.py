from PIL import Image
from torchvision import transforms


# ==========================================
# 1. CONFIGURATION
# ==========================================

# Target image size
IMG_SIZE = 224

# ImageNet normalization parameters
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ==========================================
# 2. PADDING
# ==========================================

class PadToSquare:

    def __call__(self, img):

        # Get original image dimensions
        w, h = img.size

        # Determine the square size
        side = max(w, h)

        # Create a black square image
        canvas = Image.new(img.mode, (side, side), 0)

        # Place the original image in the center
        canvas.paste(
            img,
            ((side - w) // 2, (side - h) // 2)
        )

        return canvas


# ==========================================
# 3. BASE PREPROCESSING PIPELINE
# ==========================================

eval_transform = transforms.Compose([

    # Preserve the original aspect ratio
    PadToSquare(),

    # Resize the image to 224 x 224 pixels
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Convert the image to three grayscale channels
    transforms.Grayscale(num_output_channels=3),

    # Convert the image into a PyTorch tensor
    transforms.ToTensor(),

    # Normalize using ImageNet statistics
    transforms.Normalize(mean=MEAN, std=STD)
])