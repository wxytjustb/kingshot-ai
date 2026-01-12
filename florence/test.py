#!/usr/bin/env python3
"""
Florence-2-large (0.77B) 在Mac上的完整实现
支持多种视觉任务：图像描述、目标检测、OCR等
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
from io import BytesIO

# ==================== 1. 设备检测 ====================
def get_device():
    """自动检测最佳设备"""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32  # Mac GPU必须用float32
    else:
        return "cpu", torch.float32

device, torch_dtype = get_device()
print(f"✓ 使用设备: {device} | 数据类型: {torch_dtype}")

# ==================== 2. 加载模型 ====================
print("正在加载Florence-2-large (0.77B)模型...")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large-ft",  # 0.77B参数的large版本
    trust_remote_code=True,
    torch_dtype=torch_dtype
).to(device)

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large-ft",
    trust_remote_code=True
)

print("✓ 模型加载完成")

# ==================== 3. 图像加载函数 ====================
def load_image(image_path):
    """从本地或URL加载图片"""
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return image

# ==================== 4. 推理函数 ====================
def run_inference(image, task_prompt, text_input=None):
    """
    执行推理任务
    
    参数:
        image: PIL图像对象
        task_prompt: 任务提示词（见下方任务列表）
        text_input: 某些任务需要的额外文本输入
    """
    # 构建完整prompt
    if text_input:
        prompt = task_prompt + text_input
    else:
        prompt = task_prompt
    
    # 准备输入
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(device, torch_dtype)
    
    # 生成输出
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False
    )
    
    # 解码结果
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False
    )[0]
    
    # 后处理
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    return parsed_answer

# ==================== 5. 支持的任务列表 ====================
TASKS = {
    "caption": "<CAPTION>",  # 简短描述
    "detailed_caption": "<DETAILED_CAPTION>",  # 详细描述
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",  # 超详细描述
    "object_detection": "<OD>",  # 目标检测
    "dense_region_caption": "<DENSE_REGION_CAPTION>",  # 密集区域描述
    "region_proposal": "<REGION_PROPOSAL>",  # 区域建议
    "ocr": "<OCR>",  # 文字识别
    "ocr_with_region": "<OCR_WITH_REGION>",  # 带位置的文字识别
}

# ==================== 6. 示例使用 ====================
if __name__ == "__main__":
    # 加载图片（可以是本地路径或URL）
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    # image_url = "./image.png"
    image = load_image(image_url)
    
    print("\n" + "="*60)
    print("开始执行任务...")
    print("="*60)
    
    # 示例1: 图像描述
    print("\n【任务1: 图像描述】")
    result = run_inference(image, TASKS["caption"])
    print(f"结果: {result}")
    
    # 示例2: 详细描述
    print("\n【任务2: 详细图像描述】")
    result = run_inference(image, TASKS["detailed_caption"])
    print(f"结果: {result}")
    
    # 示例3: 目标检测
    print("\n【任务3: 目标检测】")
    result = run_inference(image, TASKS["object_detection"])
    print(f"结果: {result}")
    
    # 示例4: OCR文字识别
    print("\n【任务4: 文字识别】")
    result = run_inference(image, TASKS["ocr"])
    print(f"结果: {result}")
    
    # 示例5: 带参数的任务（视觉问答）
    print("\n【任务5: 区域定位】")
    # 需要指定要定位的对象
    prompt_with_input = "<CAPTION_TO_PHRASE_GROUNDING>"
    result = run_inference(image, prompt_with_input, "hand")
    print(f"结果: {result}")
    
    print("\n" + "="*60)
    print("所有任务完成！")
    print("="*60)

# ==================== 7. 自定义任务函数 ====================
def detect_objects(image_path):
    """快捷函数：目标检测"""
    image = load_image(image_path)
    return run_inference(image, TASKS["object_detection"])

def caption_image(image_path, detailed=False):
    """快捷函数：图像描述"""
    image = load_image(image_path)
    task = TASKS["detailed_caption"] if detailed else TASKS["caption"]
    return run_inference(image, task)

def extract_text(image_path, with_region=False):
    """快捷函数：OCR文字提取"""
    image = load_image(image_path)
    task = TASKS["ocr_with_region"] if with_region else TASKS["ocr"]
    return run_inference(image, task)

# ==================== 8. 使用本地图片示例 ====================

# 使用本地图片
# local_image_path = "./image.png"
# result = detect_objects(local_image_path)
# print(result)

