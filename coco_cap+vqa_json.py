from torchvision.datasets.utils import download_url

""" json data for coco captioning """

# url = ['https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json',
# 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
# 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json']

# for u in url:
#     download_url(u,"/export/home/.cache/lavis/coco/annotations")


""" json data for coco vqa """
url = ['https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/vqa_train.json',
'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/vqa_val.json',
'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/vqa_val_eval.json',
'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/answer_list.json',
'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/v2_mscoco_val2014_annotations.json',
'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/vqa_test.json',
'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/answer_list.json']

for u in url:
    download_url(u,"/export/home/.cache/lavis/coco/annotations")