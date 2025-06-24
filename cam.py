from cv2 import cvtColor
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from torchvision.models import resnet50
from model import PointAnnoModel as create_model
import cv2
import numpy as np
import os
import torch
from torchsummary import summary

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# 1.加载模型
# model = resnet50(pretrained=True)
model = create_model('ResUNet34', 1, True)
target_layer = [model.seg_model.fc3]
# print(model.seg_model)
model = torch.nn.DataParallel(model)
model = model.cuda()
# print(model)
model_path = "/data3/zhangye/my_code_TNBC/all_experoments/experiments_TNBC/experiments_0.3/detection/MO/1.0_repeat=3/2/checkpoints/checkpoint_best.pth.tar"
best_checkpoint = torch.load(model_path)
model.load_state_dict(best_checkpoint['state_dict'])


img = cv2.imread("/data3/zhangye/my_code_TNBC/data_for_train/MO/images/test/387.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.float32(img) / 255
# img = torch.Tensor(img).cuda()
# print(img.shape)
# img = img.permute()
input_tensor = preprocess_image(img, mean=[0.7976,0.7242,0.7993],
                                             std=[0.1044,0.1295,0.0850])
print("target layer",target_layer)
cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)
grayscale_cam = cam(input_tensor=input_tensor, targets=target_layer)

grayscale_cam = grayscale_cam[0]
visualization = show_cam_on_image(img, grayscale_cam)  # (224, 224, 3)
cv2.imwrite(f'cam_dog.jpg', visualization)