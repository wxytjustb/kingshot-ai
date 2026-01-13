

# 创建环境
conda create --name yolo-env python=3.11 -y


# 激活环境
conda activate yolo-env

# 安装PyTorch（支持MPS） apple 芯片
conda install pytorch torchvision -c pytorch

# 安装YOLO（conda-forge渠道）
conda install -c conda-forge ultralytics