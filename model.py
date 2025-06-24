from email.policy import strict
import torch
import torch.nn as nn
from det_model import create_model as det_model
from seg_model import create_model as seg_model
from utils import convert_pytorch_checkpoint
import torch.nn.functional as F

class PointAnnoModel(nn.Module):
    def __init__(self, model_name, out_ch, pretrained, mode="fast", pretrained_path="../pretrained/pannuke_pretrain.tar"):#in_ch, nr_type, freeze, mode,pretrained_path="../pretrained/seg_weight.tar"):
        super().__init__()
        self.det_model = det_model(model_name, out_ch, pretrained)
        self.seg_model = seg_model(mode=mode)

        # save_state_dict = torch.load(pretrained_path)["desc"]
        # save_state_dict = convert_pytorch_checkpoint(save_state_dict)

        # self.seg_model.load_state_dict(save_state_dict, strict=False)
        # det_best_checkpoint = torch.load("/data3/zhangye/my_code_TNBC/pretrained/TNBC/checkpoint_best.pth.tar")
        # det_best_checkpoint = {k[7:]:v for k,v in det_best_checkpoint['state_dict'].items()}
        # self.det_model.load_state_dict(det_best_checkpoint, strict=False)

    def forward(self,images):
        det_out = self.det_model(images)
        seg_out = self.seg_model(images)
        seg_out.update(det_out)
        # print(F.softmax(seg_out["seg_encoding"][0]))

        return  seg_out
        # return F.softmax(seg_out["seg_encoding"][0])