# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# downstream/object_detection/convert_model_to_detectron2.py
from argparse import ArgumentParser
import pickle as pkl
from collections import OrderedDict
import torch

def convert_pth_mm():
    origin_path = r'/path/to/checkpoint'
    tar_path = r'/save/path'
    checkpoint = torch.load(origin_path)
    state_dict = checkpoint['state_dict']
    new_state = OrderedDict()
    for k in state_dict.keys():
        new_state[k.replace('backbone.', 'detector.backbone.')] = state_dict[k]
    new_weight = {'state_dict': new_state}
    torch.save(new_weight, tar_path)
    






if __name__ == "__main__":
    convert_pth_mm()
