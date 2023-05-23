import torch
from torchvision.transforms.functional import to_pil_image,resize
import src.mvtec_loader as loader

import re
import json
import numpy as np 
from lavis.models import load_model_and_preprocess
from src.util import result
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str,default='1',choices=['1','2','3','4'])
parser.add_argument('--eval_way', type=str,default='1',choices=['1','2'])
parser.add_argument('--question_way', type=str,default='1',choices=['1','2'])
args = parser.parse_args()

device=torch.device('cuda')
if args.model_type=='1': model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)
elif args.model_type=='2': model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device)
elif args.model_type=='3': model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
elif args.model_type=='4': model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)

gpu_num = torch.cuda.device_count()
if(torch.cuda.device_count()>1): 
    model = torch.nn.DataParallel(model,device_ids=np.arange(0,gpu_num)[range(0,gpu_num,1)].tolist())
"""DataParallel objectにmodelもmoduleを移す"""
model = model.module


mvtec = ['bottle','cable','carpet','hazelnut','pill','screw','toothbrush','wood','capsule','grid','leather','metal_nut','tile','transistor','zipper']
# mvtec = ['carpet','hazelnut']
result["Question"] = "Does this "+ "○○" +"hava any defect in this image? Answer:"
for breed in mvtec:
    _,test_data = loader.get_mvtec_loader(breed)
    data_num=0
    acc_num = 0
    all_path=[]
    all_label=[]
    all_pred=[]
    for idx,(data,label,_,path) in enumerate(test_data):
        pred = []
        data,label=data.to(device),label.to(device)
        if args.question_way=='1':
            Question = "Does this "+ breed +"hava any defect in this image? Answer:"
        elif args.question_way=='2':
            Question = "Does this boject hava any defect in this image? Answer:"
        # Question = "Is this wood perfect in this image? Answer:"
        text = model.generate({"image": resize(data, size=(224, 224)), "prompt": Question})
        # text = model.generate({"image": resize(data, size=(364,364)), "prompt": Question})
        
        result[breed]['path'].extend(path)
        result[breed]['pred'].extend(text)
        result[breed]['gt'].extend(list(label.cpu().numpy()))

        if args.eval_way=='1':
            for t in text:
                pred.append((np.array(re.split('[, ]',text[0]))=='yes').sum())
            pred = np.array(pred)
            pred[pred>0] = 1
            label[label>0] = 1
            pred = list(pred)
            data_num+= len(label)
            acc_num += (pred==label.cpu().numpy()).sum() 
            print(text)
            print(label)
            print('')
        elif args.eval_way=='2':
            for t in text:
                pred.append((np.array(re.split('[, ]',text[0]))=='no').sum())
                pred.append((np.array(re.split('[, ]',text[0]))=='No').sum())
            pred = np.array(pred)
            pred[pred==0] = -1; pred[pred>0] = 0
            label[label>0] = -1; label[label==0] = 0
            pred = list(pred)
            data_num+= len(label)
            acc_num += (pred==label.cpu().numpy()).sum() 
            print(text)
            print(label)
            print('')
    print(f'accuracy:{(acc_num/data_num)*100}%')
    result[breed]['test_acc'] = round((acc_num/data_num)*100,3)


import pandas as pd
df = pd.DataFrame.from_dict(result, orient='index', columns=['data'])
df.to_json('Qtype:'+args.question_way+'_type:'+args.eval_way+'_data:'+args.model_type+'.json')



            
        


