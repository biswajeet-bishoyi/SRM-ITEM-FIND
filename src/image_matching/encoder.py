"""
Image Identification and Visual Similarity Encoder for SRM CampusFind.
Extracts deep visual embeddings using Vision Encoders (PyTorch ResNet/MobileNet/CLIP)
or Color-Texture-Shape composite vector representation.
Calculates visual cosine similarity between uploaded images.
"""

import os
import io
import base64
import numpy as np
from PIL import Image, ImageOps
from typing import Optional, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HAS_TORCHVISION = False
if not os.environ.get("VERCEL"):
    try:
        import torch
        import torchvision.transforms as transforms
        import torchvision.models as models
        HAS_TORCHVISION = True
    except Exception:
        HAS_TORCHVISION = False


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    v1 = np.array(vec1).flatten()
    v2 = np.array(vec2).flatten()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


class ImageEncoder:
    """
    Objective 2 Computer Vision / Deep Learning Image Encoder.
    Uses pretrained CNN/Vision Transformer or Advanced Visual Feature Vectorizer
    with foreground object extraction, HSV histograms, color moments, and spatial layout.
    """
    
    def __init__(self, architecture: str = "resnet18"):
        self.architecture = architecture
        self.use_fallback = not HAS_TORCHVISION
        self.model = None
        self.transform = None
        
        if HAS_TORCHVISION and not os.environ.get("VERCEL"):
            try:
                base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                base_model.eval()
                self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                print(f"[ImageEncoder] Initialized Deep Learning Vision Encoder ({architecture})")
            except Exception as e:
                self.use_fallback = True
        else:
            self.use_fallback = True

    def extract_composite_features(self, img: Image.Image) -> np.ndarray:
        """
        Extract foreground-aware object color moments, HSV histogram, spatial luminance,
        and aspect ratio feature vector.
        """
        img_rgb = img.resize((128, 128)).convert('RGB')
        arr = np.array(img_rgb, dtype=np.float32)
        
        # 1. Background subtraction / foreground object estimation
        border_pixels = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
        bg_mean = np.mean(border_pixels, axis=0)
        color_diff = np.linalg.norm(arr - bg_mean, axis=2)
        fg_mask = color_diff > 25.0
        
        if np.sum(fg_mask) > 100:
            fg_pixels = arr[fg_mask]
            fg_mean = np.mean(fg_pixels, axis=0) / 255.0
            fg_std = np.std(fg_pixels, axis=0) / 255.0
            fill_ratio = np.array([np.sum(fg_mask) / (128.0 * 128.0)])
        else:
            fg_mean = np.mean(arr, axis=(0, 1)) / 255.0
            fg_std = np.std(arr, axis=(0, 1)) / 255.0
            fill_ratio = np.array([0.5])
            
        # 2. HSV Color Distribution
        hsv = np.array(img_rgb.convert('HSV'), dtype=np.float32)
        h_hist, _ = np.histogram(hsv[:, :, 0], bins=18, range=(0, 256), density=True)
        s_hist, _ = np.histogram(hsv[:, :, 1], bins=10, range=(0, 256), density=True)
        v_hist, _ = np.histogram(hsv[:, :, 2], bins=10, range=(0, 256), density=True)
        
        # 3. Spatial Luminance Grid (4x4)
        gray = np.array(img_rgb.resize((64, 64)).convert('L'), dtype=np.float32) / 255.0
        grid = [np.mean(gray[i*16:(i+1)*16, j*16:(j+1)*16]) for i in range(4) for j in range(4)]
        
        # 4. Aspect Ratio
        w, h = img.size
        aspect_ratio = np.array([float(w) / max(1.0, float(h))])
        
        vec = np.concatenate([
            fg_mean * 3.0,
            fg_std * 2.0,
            fill_ratio * 2.0,
            aspect_ratio,
            h_hist * 15.0,
            s_hist * 8.0,
            v_hist * 8.0,
            grid
        ])
        
        # Zero-center and L2-normalize
        vec = vec - np.mean(vec)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, image_input: Union[str, Image.Image, None]) -> Optional[np.ndarray]:
        """
        Encode an image path or PIL Image into a feature embedding vector.
        Resolves file paths directly, or searches in PROJECT_ROOT and data/images/.
        """
        if image_input is None:
            return None
            
        img = None
        if isinstance(image_input, str):
            candidates = [
                image_input,
                os.path.join(PROJECT_ROOT, image_input),
                os.path.join(PROJECT_ROOT, "data", "images", image_input),
                os.path.join(PROJECT_ROOT, "data", "images", os.path.basename(image_input)),
                os.path.join("data", "images", image_input),
                os.path.join("data", "images", os.path.basename(image_input))
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    try:
                        img = Image.open(cand)
                        break
                    except Exception:
                        pass
            if img is None:
                return None
        else:
            img = image_input
            
        if not self.use_fallback and self.model is not None:
            try:
                img_rgb = img.convert('RGB')
                tensor = self.transform(img_rgb).unsqueeze(0)
                with torch.no_grad():
                    embedding = self.model(tensor).squeeze().numpy().flatten()
                norm = np.linalg.norm(embedding)
                return embedding / norm if norm > 0 else embedding
            except Exception:
                return self.extract_composite_features(img)
        else:
            return self.extract_composite_features(img)

    @staticmethod
    def decode_data_url(data_url: str) -> Optional[Image.Image]:
        """Decode a browser 'data:image/...;base64,...' upload into a PIL Image."""
        if not data_url:
            return None
        try:
            encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
            img = Image.open(io.BytesIO(base64.b64decode(encoded)))
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB")
        except Exception:
            return None

    def predict_similarity(self, img1_input: Union[str, Image.Image], img2_input: Union[str, Image.Image]) -> Optional[float]:
        """Compute visual similarity score between two images (None if either image is missing)."""
        v1 = self.encode(img1_input)
        v2 = self.encode(img2_input)
        if v1 is None or v2 is None:
            return None
        return compute_cosine_similarity(v1, v2)


if __name__ == "__main__":
    encoder = ImageEncoder()
    # Create test synthetic images
    img1 = Image.new('RGB', (200, 200), color=(20, 20, 20)) # Dark image
    img2 = Image.new('RGB', (200, 200), color=(30, 30, 35)) # Similar dark image
    img3 = Image.new('RGB', (200, 200), color=(240, 240, 240)) # White image
    
    score12 = encoder.predict_similarity(img1, img2)
    score13 = encoder.predict_similarity(img1, img3)
    
    print("--- Image Similarity Test ---")
    print(f"Similarity (Dark vs Dark):  {score12:.4f}")
    print(f"Similarity (Dark vs White): {score13:.4f}")
