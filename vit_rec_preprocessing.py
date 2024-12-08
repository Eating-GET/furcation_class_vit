import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import timm

# 自定义的Transformer去噪层
class SimpleDenoiseCNN(nn.Module):
    def __init__(self):
        super(SimpleDenoiseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

class ViTWithDenoise(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', pretrained=True, num_classes=1000):
        super(ViTWithDenoise, self).__init__()
        self.denoise_cnn = SimpleDenoiseCNN()
        self.vit = timm.create_model(vit_model_name, pretrained=pretrained, num_classes=num_classes)
    
    def forward(self, x):
        x = self.denoise_cnn(x)
        x = self.vit(x)
        return x

# 数据预处理和数据增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
test_dataset = ImageFolder('./data/tooth_e/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 设置设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 模型定义
model = ViTWithDenoise(pretrained=False, num_classes=len(test_dataset.classes))
model = model.to(device)

# 加载最佳模型权重
model.load_state_dict(torch.load('./models/best_model.pth'))

# Grad-CAM
cam_extractor = SmoothGradCAMpp(model, target_layer='vit.blocks.11.attn.attn_drop')

# 创建存储Grad-CAM图像的文件夹
output_dir = './paper_writing/grad-cam-graph/'
os.makedirs(output_dir, exist_ok=True)

# 生成和保存Grad-CAM图像
model.train()  # Set the model to training mode
for idx, (inputs, targets) in enumerate(test_loader):
    inputs, targets = inputs.to(device), targets.to(device)

    # 获取模型预测
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)

    # 生成Grad-CAM图
    activation_map = cam_extractor(predicted[0].item(), outputs)
    activation_map = activation_map[0].cpu()

    # 可视化Grad-CAM图
    original_image = to_pil_image(inputs[0].cpu())
    result = overlay_mask(original_image, to_pil_image(activation_map, mode='F'), alpha=0.5)

    # 获取原始文件名并保存图像
    original_filepath = test_loader.dataset.samples[idx][0]
    filename = os.path.basename(original_filepath)
    result.save(os.path.join(output_dir, f'grad_cam_{filename}'))

    # 打印进度
    print(f'Saved Grad-CAM for image {idx + 1}/{len(test_loader)}')

print("Grad-CAM generation complete.")
