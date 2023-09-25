import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import os
import csv
from datetime import datetime
import timm

# モデルの定義と重みの読み込み
from torchvision.models import resnet18

class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 2)
        #self.feature = timm.create_model('efficientnetv2_rw_m', pretrained=True)
        #num_ftrs = self.feature.classifier.in_features
        #self.feature.classifier = nn.Linear(num_ftrs, 2)  # 2はクラスの数に合わせて変更


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h

# 入力画像の前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),       # モデルの入力サイズに合わせてリサイズ
    transforms.ToTensor(),               # Tensorに変換
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

# 入力画像の前処理 (追記: 画像の切り取りと4分割)
def preprocess_image(img):
    width, height = img.size

    # 上半分を削除
    img = img.crop((0, height // 2, width, height))

    # 右側25%を削除
    img = img.crop((0, 0, width - width // 4, img.size[1]))

    # 4分割
    quarter_width = img.size[0] // 4
    images = [
        img.crop((i * quarter_width, 0, (i + 1) * quarter_width, img.size[1])).convert('RGB')
        for i in range(4)
    ]
    return images

# 4分割した画像を推論し、結果を集計
def predict_images(images, net):
    predictions = []
    for img in images:
        img = transform(img)
        img = img.unsqueeze(0)
        y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
        predictions.append(y)
    return predictions

# テスト画像のパスを指定して分類
img_path = r'C:\Users\yuta1\OneDrive\デスクトップ\new_PLY\画像処理前\OK\7698_1PLYXXXXXX_DC56_L_OUT_OK_2023-0701-0053-27.bmp'
img = Image.open(img_path)

# 画像の前処理と4分割
processed_images = preprocess_image(img)

# ネットワークの準備
net = Net().cpu().eval()
net.load_state_dict(torch.load(r'C:\Users\yuta1\OneDrive\デスクトップ\new_PLY\new3_ply.pt', map_location=torch.device('cpu')))

# 4分割した画像を推論し、結果を集計
predictions = predict_images(processed_images, net)

# 結果の判定
if all(np.array(predictions) == 1):
    print("OK")
else:
    print("NG")