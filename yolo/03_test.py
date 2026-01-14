import ultralytics
import torch
from ultralytics import YOLO

# 检查版本
print(ultralytics.__version__)
print(torch.__version__)

# Mac用户检查MPS
print(torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    print(torch.backends.mps.is_built())

model = YOLO("/opt/homebrew/runs/detect/train4/weights/last.pt")
results = model("image1.png", conf=0.1)

results[0].show()

# results = model.predict(
#     source="./image1.png",
#     device="mps"
# )

# results[0].show()

