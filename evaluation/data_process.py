# custom/data_process.py
import os
import xml.etree.ElementTree as ET
import pandas as pd

def make_MOT_dataset():
    data_dir = r'/path/to/data/'
    class_id_map = {'nodule': 1, 'thyroid': 2}
    for video_name in os.listdir(data_dir):
        print(video_name)
        video_dir = os.path.join(data_dir, video_name)
        anno_dir = os.path.join(video_dir, 'Annotations')
        gt_file = os.path.join(video_dir, 'gt.csv')

        gt_dict = {'frame_id': [], 'instance_id': [], 'x_1': [], 'y_1': [], 'w': [], 'h': [], 'conf': [], 'class_id': [], 'visibility': []}
        anno_list = os.listdir(anno_dir)
        anno_list.sort()
        file_idx_map = dict(zip(anno_list, range(len(anno_list))))
        for xml_file in anno_list:
            frame_id = file_idx_map[xml_file]+1
            tree = ET.parse(os.path.join(anno_dir, xml_file))
            for obj_idx, anno_info in enumerate(tree.findall("object")):
                class_id = class_id_map[anno_info.findtext("name").lower()]
                instance_id = int(anno_info.find("attributes")[0][1].text) + 1
                bbox = anno_info.find("bndbox")
                x_min, y_min, x_max, y_max = float(bbox[0].text)-1, float(bbox[1].text)-1, float(bbox[2].text), float(bbox[3].text)
                w = x_max - x_min
                h = y_max - y_min
                gt_dict['frame_id'].append(frame_id)
                gt_dict['instance_id'].append(instance_id)
                gt_dict['x_1'].append(int(x_min))
                gt_dict['y_1'].append(int(y_min))
                gt_dict['w'].append(int(w))
                gt_dict['h'].append(int(h))
                conf = 1 if class_id == 1 else 0
                gt_dict['conf'].append(conf)
                gt_dict['class_id'].append(class_id)
                gt_dict['visibility'].append(1)
        df = pd.DataFrame(gt_dict)
        df.sort_values(['instance_id','frame_id'], axis=0, ascending=True, inplace=True)
        df.to_csv(gt_file, header=None, index=False)


if __name__  == '__main__':
    make_MOT_dataset()

