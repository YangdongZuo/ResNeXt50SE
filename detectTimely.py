import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# 加载模型并设置最后一层
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 二分类
model.load_state_dict(torch.load('model_resnet50se.pth'))
model.eval()

# 类别名称
class_names = ['bing', 'zu']

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 打开摄像头
cap = cv2.VideoCapture(0)

try:
    while True:
        # 捕获视频帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 将图像从 BGR 转换为 RGB，然后应用预处理
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)

        # 进行预测
        with torch.no_grad():
            out = model(batch_t)
        _, predicted = torch.max(out, 1)
        prediction = class_names[predicted.item()]

        # 在图像上显示预测结果
        cv2.putText(frame, f'Predicted: {prediction}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Camera', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
