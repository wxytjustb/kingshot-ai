from roboflow import Roboflow

rf = Roboflow(api_key="mNWYXKcmV8NinEqPp5xB")
project = rf.workspace("rokywang").project("game-wlr4k")

# 下载YOLOv8格式（兼容YOLO11）
dataset = project.version(2).download("yolov11")