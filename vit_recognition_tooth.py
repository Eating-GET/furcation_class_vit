import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler

# 数据预处理和数据增强
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁剪和调整大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ColorJitter(),  # 随机颜色变换
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载数据集
train_dataset = ImageFolder('./data/tooth_e/train', transform=transform)
test_dataset = ImageFolder('./data/tooth_e/test', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 设置设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 模型定义
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(train_dataset.classes))
model = model.to(device)

# 使模型的更多层可训练
for param in model.parameters():
    param.requires_grad = True

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 使用L2正则化

# 学习率调度器
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 训练和验证函数
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == targets).sum().item()
    accuracy = total_correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

def validate_epoch(model, loader, device):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == targets).sum().item()
    accuracy = total_correct / len(loader.dataset)
    return accuracy

# 执行训练
epochs = 50
best_accuracy = 0
for epoch in range(epochs):
    train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
    test_accuracy = validate_epoch(model, test_loader, device)
    scheduler.step()  # 更新学习率
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    # 保存最佳模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), './models/best_model.pth')

print("Training complete. Best test accuracy: {:.4f}".format(best_accuracy))
