import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from ResNet50SE import ResNet50SE, SEBottleneck

# 数据预处理
def get_data_loaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(data_dir + '/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = datasets.ImageFolder(data_dir + '/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    test_dataset = datasets.ImageFolder(data_dir + '/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

# 模型定义
def initialize_model(num_classes=2, groups=32):
    model = ResNet50SE(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, groups=groups)
    return model

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 验证函数
def validate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')
    return accuracy

# 主函数
def main():
    data_dir = 'data'  # 路径需根据实际情况调整
    train_loader, val_loader, test_loader = get_data_loaders(data_dir)

    model = initialize_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, num_epochs=10)
    validate_model(model, val_loader)
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved as trained_model.pth")

    print("Testing model...")
    validate_model(model, test_loader)

if __name__ == '__main__':
    main()
