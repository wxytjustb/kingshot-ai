from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")

# 使用MPS训练（利用M1/M2/M3/M4 GPU）
results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    batch=8,          # 小批次
    imgsz=640,
    patience=20,      # 验证集20轮不提升就停止
    dropout=0.2,      # 增加dropout
    lr0=0.001,        # 较小学习率
    weight_decay=0.0005,  # 权重衰减
    device="mps"  # 关键：指定MPS设备
)
