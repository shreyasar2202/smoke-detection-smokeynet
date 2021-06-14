import pickle
from collections import defaultdict
from torch.utils.data.dataset import Dataset
from pathlib import Path
import numpy as np

class TiledDataloader(Dataset):
    def __init__(self, data_dir, img_list=None, series_length=5, transform=None):
        self.data_dir = Path(data_dir)
        with (self.data_dir/'prev_imgs_dict.pkl').open('rb') as pkl_file:
            self.prev_imgs = pickle.load(pkl_file)
        self.all_imgs = img_list or list(self.prev_imgs)
        self.series_length = series_length
        self.transform = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        all_imgs = []
        cur_filename = self.all_imgs[idx]
        for _ in range(self.series_length):
            all_imgs.append(np.load(self.data_dir/f'{cur_filename}_img.npy'))
            cur_filename = self.prev_imgs[cur_filename]
        # Want it to be GRID_IDX, TIMESTEP, CHANNEL, H, W
        all_imgs = np.transpose(np.stack(all_imgs), (1, 0, 2, 3, 4))/255
        all_labels = np.load(self.data_dir/f'{cur_filename}_lbl.npy')
        cur_item = {"image": all_imgs, "class": all_labels}
        if self.transform is not None:
            cur_item = self.transform(cur_item)
        return cur_item