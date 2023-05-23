import torch
import torch.nn as nn
import src.mvtec_loader as loader

import numpy as np 
from lavis.models import load_model_and_preprocess
import argparse
from src.mvtec_loader import get_mvtec_loader
# from lavis.models.


# parser = argparse.ArgumentParser()
# parser.add_argument('--model_type', type=str,default='1',choices=['1','2','3','4'])
# parser.add_argument('--eval_way', type=str,default='1',choices=['1','2'])
# parser.add_argument('--question_way', type=str,default='1',choices=['1','2'])
# args = parser.parse_args()

device=torch.device('cuda')
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)

gpu_num = torch.cuda.device_count()
if(torch.cuda.device_count()>1): 
    model = torch.nn.DataParallel(model,device_ids=np.arange(0,gpu_num)[range(0,gpu_num,1)].tolist())
"""DataParallel objectにmodelもmoduleを移す"""
model = model.module


# class Qformer():
#     def __init__(self,model):
#         super(Qformer, self).__init__()

#     def forward(self,feature):




_,test = get_mvtec_loader('bottle')
vision_encoder = model.visual_encoder.float()
# text_encoder = model.text_encoder.float()
qformer = model.Qformer
vision_criterion = nn.CrossEntropyLoss(weight=None).to(device)
text_criterion = nn.CrossEntropyLoss(weight=None).to(device)

for idx,(data,label,_,path) in enumerate(test):
    data,_ = data.to(device),label.to(device)
    visual_enbedding = vision_encoder(data)
    import pdb;pdb.set_trace()


# model_0.visual_encoder()
# model = model_0.Qformer
# model.train()

