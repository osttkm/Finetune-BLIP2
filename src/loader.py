from torch.utils.data import Dataset
import numpy as np
import torch

class vlm_dataset(Dataset):
    def __init__(self, image_path, caption_path, vocab):
        self.images = np.load(image_path)
        self.captions = np.load(caption_path)
        self.vocab = vocab

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(self.images[index])
        caption = self.captions[index]
        caption = [self.vocab(token) for token in caption.split()]
        caption = torch.tensor(caption)
        return image, caption

class vlm_loader(Dataset):