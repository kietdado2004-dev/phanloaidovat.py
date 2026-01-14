import os
import torch
import warnings
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights

warnings.filterwarnings("ignore")

# ================== FONT TIẾNG VIỆT ==================
plt.rcParams["font.family"] = "DejaVu Sans"

# ================== THƯ MỤC ẢNH ==================
IMAGE_DIR = "images"

image_paths = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_paths:
    raise RuntimeError("❌ Thư mục images không có ảnh")

# ================== MODEL ==================
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()

transform = weights.transforms()
labels = weights.meta["categories"]

# ================== NHÓM PHÂN LOẠI ==================
AO = ["t-shirt", "shirt", "jacket", "coat", "sweater"]
QUAN = ["jean", "jeans", "pants", "trousers"]

# ================== HIỂN THỊ ==================
plt.figure(figsize=(12, 3.5 * len(image_paths)))

for idx, path in enumerate(image_paths):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        y = model(x)
        prob = torch.softmax(y, dim=1)
        top5 = torch.topk(prob, 5)

    # ===== KẾT LUẬN =====
    ket_luan = "ĐỒ VẬT"
    for i in top5.indices[0]:
        name = labels[i].lower()
        if any(k in name for k in AO):
            ket_luan = "ÁO"
            break
        if any(k in name for k in QUAN):
            ket_luan = "QUẦN"
            break

    # ===== BỐ CỤC ẢNH + CHỮ (GẦN NHAU) =====
    ax_img = plt.subplot(len(image_paths), 2, idx * 2 + 1)
    ax_txt = plt.subplot(len(image_paths), 2, idx * 2 + 2)

    # ẢNH
    ax_img.imshow(img)
    ax_img.axis("off")

    # CHỮ
    ax_txt.axis("off")

    ax_txt.text(
        0.02, 0.55,
        "KẾT LUẬN:",
        fontsize=16,
        weight="bold",
        ha="left",
        va="center"
    )

    ax_txt.text(
        0.02, 0.35,
        ket_luan,
        fontsize=28,
        weight="bold",
        ha="left",
        va="center"
    )

plt.tight_layout(w_pad=1.5)
plt.show()
