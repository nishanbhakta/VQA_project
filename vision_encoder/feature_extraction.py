# pip install torch torchvision pillow requests
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
import requests
import gc

# 1) Image loading utilities

def load_image_from_path(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')

def load_image_from_url(url: str) -> Image.Image:
    r = requests.get(url, stream=True, timeout=10)
    r.raise_for_status()
    return Image.open(r.raw).convert('RGB')

def load_image_from_bytes(b: bytes) -> Image.Image:
    return Image.open(BytesIO(b)).convert('RGB')

# 2) Preprocessing

IMG_SIZE = 224
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    return preprocess(pil_img).unsqueeze(0)  # (1,3,224,224)

# -------------------------------
# 3) Document / Non-document router

class DocRouter(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, 2)

    def forward(self, x):
        return self.backbone(x)

@torch.no_grad()
def is_document(model: DocRouter, x: torch.Tensor, thresh=0.5) -> bool:
    with torch.amp.autocast("cuda", enabled=x.device.type == "cuda"):
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        doc_prob = probs[:, 1]
    return bool(doc_prob.item() >= thresh)

# -------------------------------
# 4) Image encoder: ResNet50 -> 768
# -------------------------------
class ResNetEncoder(nn.Module):
    def __init__(self, out_dim=768, pretrained=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
        self.proj = nn.Linear(2048, out_dim)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=x.device.type == "cuda"):
            feats = self.cnn(x).flatten(1)
            out = self.proj(feats)
        return out

# -------------------------------
# 5) Utility: free GPU memory
# -------------------------------
def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()

# -------------------------------
# 6) Main classify + embed function
# -------------------------------
def classify_and_embed(pil_img, device="cuda"):
    x = preprocess_image(pil_img).to(device)

    # ----- Step 1: Router -----
    router = DocRouter(pretrained=True).to(device)
    router.eval()
    try:
        is_doc_flag = is_document(router, x)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device == "cuda":
            print("⚠️ GPU OOM in router. Switching to CPU...")
            free_gpu()
            device = "cpu"
            router.to("cpu")
            x = x.to("cpu")
            is_doc_flag = is_document(router, x)
        else:
            raise e
    # Free router GPU memory
    del router
    free_gpu()

    # ----- Step 2: Encoder -----
    img_encoder = ResNetEncoder(out_dim=768, pretrained=True).to(device)
    img_encoder.eval()
    try:
        img_vec = img_encoder(x)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device == "cuda":
            print("⚠️ GPU OOM in encoder. Switching to CPU...")
            free_gpu()
            device = "cpu"
            img_encoder.to("cpu")
            x = x.to("cpu")
            img_vec = img_encoder(x)
        else:
            raise e
    finally:
        del img_encoder
        free_gpu()

    label = "document" if is_doc_flag else "non_document"
    return label, img_vec

# -------------------------------
# 7) Example usage
# -------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Initial device:", device)

    pil_img = load_image_from_path("chart.png")  # change to your image path

    label, img_vec = classify_and_embed(pil_img, device=device)
    print("\nClass:", label)
    print("Embedding shape:", img_vec.shape)
    print("Embedings: ", img_vec)
