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

try:
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as models
    HAS_TORCHVISION = True
except ImportError:
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
    Uses pretrained CNN/Vision Transformer or PIL Visual Feature Vectorizer
    to extract shape, color histogram, texture, and structural deep features.
    """
    
    def __init__(self, architecture: str = "resnet18"):
        self.architecture = architecture
        self.use_fallback = not HAS_TORCHVISION
        self.model = None
        self.transform = None
        
        if HAS_TORCHVISION:
            try:
                # Initialize ResNet18 feature extractor (removing final classification fc layer)
                base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                base_model.eval()
                # Remove last FC layer to get 512-dim feature embedding
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
                print(f"[ImageEncoder] Warning loading PyTorch model ({e}). Using Composite Visual Feature Extractor.")
                self.use_fallback = True
        else:
            print("[ImageEncoder] PyTorch/torchvision not detected. Using Composite Visual Feature Extractor.")

    def extract_composite_features(self, img: Image.Image) -> np.ndarray:
        """
        Extract non-pixel-wise spatial RGB color histogram + aspect ratio + texture feature vector.
        Guarantees fast, robust visual matching without needing large GPU downloads.
        """
        img_resized = img.resize((128, 128)).convert('RGB')
        arr = np.array(img_resized)
        
        # 1. 3D Color Histograms (RGB channels, 16 bins each = 48 features)
        r_hist, _ = np.histogram(arr[:, :, 0], bins=16, range=(0, 256), density=True)
        g_hist, _ = np.histogram(arr[:, :, 1], bins=16, range=(0, 256), density=True)
        b_hist, _ = np.histogram(arr[:, :, 2], bins=16, range=(0, 256), density=True)
        
        # 2. Aspect Ratio feature
        w, h = img.size
        aspect_ratio = np.array([w / max(1, h)])
        
        # 3. Spatial Grid Luminance Intensity (4x4 spatial blocks = 16 features)
        gray = img_resized.convert('L')
        gray_arr = np.array(gray)
        grid_features = []
        h_chunk, w_chunk = 32, 32
        for i in range(4):
            for j in range(4):
                block = gray_arr[i*h_chunk:(i+1)*h_chunk, j*w_chunk:(j+1)*w_chunk]
                grid_features.append(np.mean(block) / 255.0)
                grid_features.append(np.std(block) / 255.0)
                
        feature_vector = np.concatenate([r_hist, g_hist, b_hist, aspect_ratio, grid_features])
        # Normalize
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm
        return feature_vector

    def encode(self, image_input: Union[str, Image.Image, None]) -> Optional[np.ndarray]:
        """
        Encode an image path or PIL Image into a feature embedding vector.
        Returns None when no image is available, so the fusion engine can treat
        the visual modality as missing instead of comparing meaningless vectors.
        """
        if image_input is None:
            return None
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return None
            img = Image.open(image_input)
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
            except Exception as e:
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
