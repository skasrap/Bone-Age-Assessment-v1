# models/gender_model.py
import torch
import torch.nn as nn
from torchvision import models

class Gender_Model(nn.Module):
    def __init__(self, dense_layer1_size, dense_layer2_size, p_dropout, regressor_layer_size):
        super(Gender_Model, self).__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V2')
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.new_conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.new_conv1.weight = nn.Parameter(self.resnet.conv1.weight.mean(dim=1, keepdim=True))
        self.resnet.conv1 = self.new_conv1

        self.gender_dense = nn.Sequential(
            nn.Linear(1, dense_layer1_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout),
            nn.Linear(dense_layer1_size, dense_layer2_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout)
        )

        self.fc1 = nn.Linear(dense_layer2_size + num_features, regressor_layer_size)
        self.fc2 = nn.Linear(regressor_layer_size, 1)

    def forward(self, img, gender):
        img_features = self.resnet(img)
        gender_features = self.gender_dense(gender.unsqueeze(1))
        combined_features = torch.cat((img_features, gender_features), dim=1)
        x = self.fc1(combined_features)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x
