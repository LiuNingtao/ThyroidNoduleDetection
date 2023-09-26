# solo/utils/custom_dataset.py
import os
import re
from statistics import mode
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import math
import pandas as pd
from torchvision import transforms

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def load_annotations(data_prefix):
    
    samples = []
    for root, _, file_list in os.walk(data_prefix):
        file_list = list(filter(lambda fn: has_file_allowed_extension(fn, ('.PNG', '.png', '.tif', '.TIF', '.bmp')), file_list))
        for file_name in file_list:
            samples.append(os.path.join(root, file_name))
    data_infos = []
    for i, filename in enumerate(samples):
        info = {'img_prefix': None}
        info['img_info'] = {'filename': filename}
        info['gt_label'] = np.array([0], dtype=np.int64)
        info['idx'] = int(i)
        data_infos.append(info)
    return data_infos

class BaseTransform:
    """Adds callable base class to implement different transformation pipelines."""

    def __call__(self, x: Image) -> torch.Tensor:
        return self.transform(x)

    def __repr__(self) -> str:
        return str(self.transform)

class CustomDataset(Dataset):
    def __init__(self, data_prefix, transformer):
        super(CustomDataset, self).__init__()
        self.data_infos = load_annotations(data_prefix)
        self.ps_transormer = PatchDisorder(n_iter=25, 
                              scales=[], 
                              ratios=[],
                              scale_p=[],
                              ratio_p=[]
                              )
        self.transformer = transformer
        reso_path = r'/path/to/resolution.csv'
        reso_dict = pd.read_csv(reso_path).to_dict(orient='list')
        self.reso_dict = dict(zip(list(map(lambda x: x.replace('.bmp', ''), reso_dict['NAME'])), reso_dict['RESO']))
        

    def __getitem__(self, index):
        file_name = self.data_infos[index]['img_info']['filename']
        v_name = file_name.split('/')[-3]
        reso = self.reso_dict[v_name]
        target = self.data_infos[index]['gt_label'][0]

        with open(file_name, 'rb') as f:
            img = Image.open(f)
            img =  img.convert('RGB')
        img = self.ps_transormer(img, reso=reso)
        img = self.transformer(img)
        return img, torch.tensor(target)
    
    def __len__(self):
        return len(self.data_infos)

class ReconstructDataset(Dataset):
    def __init__(self, data_prefix, transformer) -> None:
        # print(data_prefix)
        super().__init__()
        self.data_prefix = data_prefix
        file_path_list = []
        for root, dirs, file_list in os.walk(data_prefix):
            for file_name in file_list:
                if file_name.endswith('.PNG'):
                    file_path_list.append(os.path.join(root, file_name))
        self.file_list = file_path_list
        self.transformer = transformer
    
    def __getitem__(self, index):
        file_path = os.path.join(self.data_prefix, self.file_list[index])
        image = Image.open(file_path)
        W, H = 448, 448
        image_input = self.transformer(image)
        image_scaled_list = [image.resize(size=(W//scale, H//scale)) for scale in [1, 2, 4, 8]]
        image_scaled_list = list(map(lambda x: torch.tensor(np.array(x)/255, dtype=torch.float32), image_scaled_list))
        return image_input, *image_scaled_list

    def __len__(self):
        return len(self.file_list)

class PatchDisorder(object):
    """ patch disorder augmentation refers to 'ContextReorder'

    Args:
        n_iter(int), the times of selecting and swaping the selected patch-pair.
        scales(List(int)): The scales of selected patch.
        ratios(List(float), optional): The ratios of selected patch.
            Defaults to (1,)
    """
    def __init__(self, n_iter, scales, ratios=(1,), scale_p=None, ratio_p=None):
        self.n_iter = n_iter
        self.scales = scales
        self.ratios = ratios
        if scale_p and np.sum(scale_p) != 1:
            scale_p = np.array(scale_p)
            scale_p = self.generate_proportions(scale_p)
            scale_p[-1] = scale_p[-1] - 0.001
            print(scale_p)
        if ratio_p and np.sum(ratio_p) != 1:
            ratio_p = np.array(ratio_p)
            ratio_p = self.generate_proportions(ratio_p)
            print(ratio_p)
        self.scale_p = scale_p
        self.ratio_p = ratio_p
    
    def __call__(self, img, reso=None):
        img_mode = img.mode
        img = np.array(img, dtype=np.uint8)
        H, W = img.shape[:2]
        for i in range(self.n_iter):
            scale = np.random.choice(self.scales, 1, replace=True, p=self.scale_p)[0]
            if reso:
                scale = scale / reso
            scale = np.sqrt(scale)
            ratio = np.random.choice(self.ratios, 1, replace=True, p=self.ratio_p)[0]
            ratio = np.sqrt(ratio)
            h_patch = scale * ratio
            if scale > (W/2) or h_patch > (H/2):
                scale = scale / math.ceil((scale/W)*2)
                h_patch = h_patch / math.ceil((h_patch/H)*2)
            x_selected = np.random.choice(int(W-scale), 2, replace=True)
            y_selected = np.random.choice(int(H-h_patch), 2, replace=True)

            if self._is_close(x_selected, y_selected, scale, ratio):
                continue

            patch_1 = img[y_selected[0]: int(y_selected[0]+h_patch), x_selected[0]: int(x_selected[0]+scale), :]
            patch_2 = img[y_selected[1]: int(y_selected[1]+h_patch), x_selected[1]: int(x_selected[1]+scale), :]
            img[y_selected[1]: int(y_selected[1]+h_patch), x_selected[1]: int(x_selected[1]+scale), :] = patch_1
            img[y_selected[0]: int(y_selected[0]+h_patch), x_selected[0]: int(x_selected[0]+scale), :] = patch_2
            # img = np.array(img, dtype=np.uint8)
            # Image.fromarray(img, mode=img_mode).save('/home/ROBARTS/nliu/Project/ThyroidNodule/solo-learn/disordered.png')
        return Image.fromarray(img, mode=img_mode)
    @staticmethod
    def _is_close(x_selected, y_selected, scale, ratio):
        dis_select = (x_selected[0] - x_selected[1]) ** 2 + (y_selected[0] - y_selected[1]) ** 2
        dis_patch = scale ** 2 + int(scale*ratio) ** 2
        if dis_select < dis_patch:
            return True
        return False
    @staticmethod
    def generate_proportions(input_array):
    # Ensure input_array is a 1-D numpy array
        if input_array.ndim != 1:
            raise ValueError("Input array must be 1-D")

        # Normalize input_array
        normalized_array = input_array / np.sum(input_array)

        # Calculate scaling factor to ensure the sum is 1
        scaling_factor = 1 / np.sum(normalized_array)

        # Scale the normalized_array to satisfy the sum condition
        output_array = normalized_array * scaling_factor

        # Round the elements to two decimal places
        output_array = np.round(output_array, 3)

        return output_array





