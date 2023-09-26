# custom/data_process.py
from copy import copy
import torch
import json
import os
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pycocotools.coco import COCO
from mmtrack.datasets import CocoVID
import random
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

def load_pth():
    path = r'/path/to/pkl'
    weight = torch.load(path)['state_dict']
    pass

def load_pkl():
    path = r'/path/to/pkl'
    data = np.load(path, allow_pickle=True)
    print(len(data))

def half_train():
    origin_path = r'/path/to/train_cocoformat.json'
    origin_anno = json.load(open(origin_path, mode='r'))
    video_list = origin_anno['videos'].copy()
    video_list = list(filter(lambda x: int(x['id']) <= 12, video_list))
    image_list = origin_anno['images'].copy()
    image_list = list(filter(lambda x: int(x['video_id']) <= 12, image_list))
    # anno_list = origin_anno['annotations'].copy()
    # anno_list = list(filter(lambda x: int(x['video_id']) % 2 != 0, anno_list))
    origin_anno['videos'] = video_list
    origin_anno['images'] = image_list
    # origin_anno['annotations'] = anno_list
    json.dump(origin_anno, open(origin_path.replace('train_cocoformat.json', 'save_name.json'), mode='w'))

def convert_coco_from_xml():
    xml_path = r'/path/to/xml/annotation'
    save_dir = r'/save/path'
    tree = ET.parse(xml_path)
    root = tree.getroot()
    video_dict = {}
    for task in root.find('meta').find('project').find('tasks').findall('task'):
        id = task.find('id').text
        name = task.find('name').text
        video_dict[name] = id
    
    
    video_json = r'/path/to/videos.json'
    video_json = json.load(open(video_json, mode='r'))
    train_list = video_json['train']
    test_list = video_json['test']
    video_list = video_json['videos']

    train_list = list(map(lambda x: x.replace('*', '').replace(' ', ''), train_list))
    test_list = list(map(lambda x: x.replace('*', '').replace(' ', ''), test_list))
    video_list = list(map(lambda x: x.replace('*', '').replace(' ', ''), video_list))
    
    assert len(set(train_list).intersection(set(test_list))) == 0
    assert set(train_list).union(set(test_list)) == set(video_list)
    
    image_id_begin = 1
    box_id_begin = 1
    image_all = []
    bbox_all = []
    video_list = []
    categories = [
        {
            "id": 1,
            "name": "nodule"
        },
        {
            "id": 2,
            "name": "thyroid"
        }
    ]
    for idx, video in enumerate(train_list):
        video_id = idx + 1
        image_list, box_list, image_id_begin, box_id_begin = process_per_video(root, video_dict[video], video_id, image_id_begin, box_id_begin, video)
        image_all.extend(image_list)
        bbox_all.extend(box_list)
        video_list.append(
            {'id': video_id, 'name': video}
        )

    train_anno = {
        'categories': categories,
        'videos': video_list,
        'images': image_all,
        'annotations': bbox_all
    }
    json.dump(train_anno, open(os.path.join(save_dir, 'train.json'), mode='w', encoding='utf8'))


   
def process_per_video(root, video_id, video_idx, image_id_begin, box_id_begin, task_name=None):
    attribute = f'.//image[@task_id="{video_id}"]'
    images = root.findall(attribute)
    box_list = []
    image_list = []
    frame_id = 0
    images.sort(key=lambda x: x.attrib['name'])
    for image in images:
        id = image.get('id')
        name = image.get('name')
        width = image.get('width')
        height = image.get('height')
        for box in image.findall('box'):
            label = box.get('label')
            occluded = int(box.get('occluded'))
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            bbox = [xtl, ytl, xbr-xtl, ybr-ytl]
            box_list.append(dict(
                id = box_id_begin,
                image_id = image_id_begin,
                video_id = video_idx,
                category_id = 1 if label == 'NODULE' else 2,
                bbox = bbox,
                area = bbox[2] * bbox[3],
                occluded = False if occluded == 0 else True,
                truncated = False,
                iscrowd = False,
                ignore = False,
                is_vid_train_frame = False,
                visibility = 1.0
            ))
            box_id_begin += 1
        image_list.append(dict(
            file_name = task_name+'/'+name,
            height = float(height),
            width = float(width),
            id = image_id_begin,
            video_id = video_idx,
            frame_id = frame_id
        ))
        image_id_begin += 1
        frame_id += 1
    return image_list, box_list, image_id_begin, box_id_begin

def move_image():
    xml_path = r'/path/to/xml/annotation'
    save_dir = r'/path/to/images'
    tree = ET.parse(xml_path)
    root = tree.getroot()
    video_dict = {}
    for task in root.find('meta').find('project').find('tasks').findall('task'):
        id = task.find('id').text
        name = task.find('name').text
        os.makedirs(os.path.join(save_dir, name), exist_ok=True)
        video_dict[id] = name
    attribute = f'.//image'
    images = root.findall(attribute)
    for image in tqdm(images):
        video_name = video_dict[image.attrib['task_id']]
        image_name = image.attrib['name']
        shutil.move(os.path.join(save_dir, 'default', image_name), 
        os.path.join(save_dir, video_name, image_name))


def init_new_video(video_path):
    image_list = os.listdir(os.path.join(video_path, 'JPEGImages'))
    image_list.sort()
    image_map = dict(zip(image_list, [format(i+1, '06d') for i in range(len(image_list))]))
    for file_name in image_list:
        # assert file_name.startswith('IM-')
        os.rename(os.path.join(video_path, 'JPEGImages', file_name), os.path.join(video_path, 'JPEGImages', image_map[file_name]+'.tif'))
        tree = ET.parse(os.path.join(video_path, "Annotations", file_name.replace('.tif', '.xml')))
        tree.find('filename').text = image_map[file_name]+'.tif'
        tree.write(os.path.join(video_path, "Annotations", image_map[file_name]+'.xml'))
        os.remove(os.path.join(video_path, "Annotations", file_name.replace('.tif', '.xml')))
    print('DONE')


def init_video_set():
    root_dir = r'/path/to/data/rootdata'
    for task_name in []:
        if task_name.startswith('task_'):
            print(task_name)
            init_new_video(os.path.join(root_dir, task_name))

def convert_pas_coco():
    is_train_test = False
    root_dir = r'/path/to/data/rootdata'
    if not is_train_test:
        save_path = r'/path/to/data/'
    else:
        save_path = r'/path/to/data/rootdataset/'
    fold_json = open(r'/path/to/data/root/')
    fold_json = json.load(fold_json)
    video_fold_dict = fold_json['fold']
    cate_list = [{"id": 1, "name": "nodule"}, {"id": 2, "name": "thyroid"},]
    category_map_dict = {"nodule": 1, 'NODULE': 1, 'thyroid': 2, 'THYROID': 2}
    all_video = os.listdir(root_dir)
    all_video = list(filter(lambda x: not x.startswith('.'), all_video))
    if not is_train_test:
        fold_num = 5
        iter_list = list(range(1, 6))
    else:
        iter_list = ['train_test_filtered',]

    for fold in iter_list:
        # test_begin, test_end = fold * fold_length, (fold + 1) * fold_length
        # dataset = {'train': all_video[:test_begin]+all_video[test_end:], 'val': all_video[test_begin:test_end], 'test':  all_video[test_begin:test_end]}
        if not is_train_test:
            train_all = list(filter(lambda x: video_fold_dict[x] != fold, list(video_fold_dict.keys())))
            test_all = list(filter(lambda x: video_fold_dict[x] == fold, list(video_fold_dict.keys())))
            dataset = {'train': train_all, 'test': test_all}
        else:
            train_all, test_all, val_all = get_dataset_list(fold_json)
            dataset = {'train': train_all, 'test': test_all, 'val': val_all}
        all_nodule_num = 0
        for key in dataset.keys():
            video_id = 0
            image_id = 0
            anno_id = 0
            current_instance_dict = {}
            image_list = []
            annotation_list = []
            video_list = []
            sequence_list = dataset[key]
            for idx_1, video in enumerate(sequence_list):
                print(video)
                video_id += 1
                video_list.append({"id": video_id, "name": video})
                
                instance_id = 0
                anno_list = os.listdir(os.path.join(root_dir, video, "Annotations"))
                anno_list.sort()
                for idx_2, anno_name in enumerate(anno_list):
                    tree = ET.parse(os.path.join(root_dir, video, "Annotations", anno_name))
                    root = tree.getroot()
                    file_name = root[1].text.split('-')[-1]
                    height, width = int(root.findtext('size/height')), int(root.findtext('size/width'))
                    image_id += 1
                    image_list.append({"file_name": video+'/' + 'JPEGImages/' + anno_name.replace('.xml', '.tif'), "height": height, "width": width, "id": image_id, "video_id": idx_1+1, "frame_id": idx_2})
                    for idx_3, anno_info in enumerate(tree.findall("object")):
                        category_name = anno_info.findtext("name")
                        if category_name not in category_map_dict.keys():
                            continue
                        category_id = category_map_dict[category_name]
                        anno_id += 1
                        try:
                            track_id = video + '_' + anno_info.find("attributes")[0][1].text
                        except:
                            track_id = video + '_' + str(0)
                        if not track_id in current_instance_dict.keys():
                            instance_id += 1
                            current_instance_dict[track_id] = instance_id
                            if category_id == 1:
                                all_nodule_num += 1
                        track_id = current_instance_dict[track_id]
                        bbox = anno_info.find("bndbox")
                        x_min, y_min, x_max, y_max = float(bbox[0].text)-1, float(bbox[1].text)-1, float(bbox[2].text), float(bbox[3].text)
                        assert x_max > x_min and y_max > y_min
                        bbox = [x_min, y_min, x_max-x_min, y_max-y_min]
                        annotation_list.append({"id": anno_id,
                                                "image_id": image_id,
                                                "video_id": video_id,
                                                "category_id": category_id,
                                                "instance_id": track_id,
                                                "bbox": bbox,
                                                "area": bbox[2] * bbox[3],
                                                "occluded": False,
                                                "truncated": False,
                                                "iscrowd": False,
                                                "ignore": False,
                                                "is_vid_train_frame": False,
                                                "visibility": 1.0})
            if not os.path.exists(os.path.join(save_path, str(fold))):
                os.makedirs(os.path.join(save_path, str(fold)))
            json_file = open(os.path.join(save_path, str(fold), '{}.json'.format(key)), mode='w', encoding='utf8')
            json.dump({"categories": cate_list, "videos": video_list, "images": image_list, "annotations": annotation_list}, json_file)

def get_dataset_list(fold_json):
    all_patient = [x for x in list(fold_json['number'].keys())]
    train_list, test_list = train_test_split(all_patient, test_size=0.4)
    test_list, val_list = train_test_split(test_list, test_size=0.5)
    train_all = []
    test_all = []
    val_all = []
    for video_name in fold_json['list']:
        patient_name = video_name.split('-')[0]
        if patient_name in train_list:
            train_all.append(video_name)
        elif patient_name in test_list:
            test_all.append(video_name)
        elif patient_name in val_list:
            val_all.append(video_name)
        else:
            print('ERROR')
            print(video_name)
    return train_all, test_all, val_all
    

def instance_length_statistic():
    anno_dir = r'/path/to/data/'
    instance_dict = {}

    for fold in range(1, 6):
        fold = str(fold)
        json_file = open(os.path.join(anno_dir, fold, 'test.json'), mode='r')
        anno_json = json.load(json_file)
        anno_list = anno_json['annotations']
        for anno in anno_list:
            if anno['category_id'] == 1:
                instance_id = fold + '_' + str(anno['video_id']) + '_' + str(anno['instance_id'])
                if instance_id not in instance_dict.keys():
                    instance_dict[instance_id] = 0
                instance_dict[instance_id] += 1
    
    len_list = []
    for key in instance_dict.keys():
        len_list.append(instance_dict[key])
    pd.DataFrame({'INSTANCE_LEN': len_list}).to_csv(r'/path/to/data/instance_len_dis.csv', index=False)


def sample_frames():
    root_dir = r'/path/to/data/rootdata/'
    save_dir = r'/save/path/'
    frame_gap = 3

    for file_name in os.listdir(root_dir):
        anno_path = os.path.join(root_dir, file_name)
        anno_json = json.load(open(anno_path, mode='r'))
        images = anno_json['images']
        annos = anno_json['annotations']

        sampled_images = []
        
        new_id_begin = 1
        id_map_dict = {}
        frame_id_dict = {}
        for image in images:
            old_id =  image['id']
            if old_id % frame_gap == 0:
                image_new = copy.deepcopy(image)
                image_new['id'] = new_id_begin
                id_map_dict[old_id] = new_id_begin
                sampled_images.append(image_new)
                if image_new['video_id'] not in frame_id_dict:
                    frame_id_dict[image_new['video_id']] = -1
                frame_id_dict[image_new['video_id']] += 1
                image_new['frame_id'] = frame_id_dict[image_new['video_id']]

                new_id_begin += 1
        
        new_id_begin_anno = 1
        sampled_annos = []
        for anno in annos:
            image_id = anno['image_id']
            if image_id in id_map_dict:
                new_anno = copy.deepcopy(anno)
                new_anno['image_id'] = id_map_dict[image_id]
                new_anno['id'] = new_id_begin_anno
                new_id_begin_anno += 1
                sampled_annos.append(new_anno)
        anno_json['images'] = sampled_images
        anno_json['annotations'] = sampled_annos
        json.dump(anno_json, open(os.path.join(save_dir, file_name), mode='w'))


def statistic_size_dis():
    anno_dir = r'/path/to/data/rootdata/'
    reso_path = r'/path/to/data/rootresolution.csv'
    reso_dict = pd.read_csv(reso_path).to_dict(orient='list')
    reso_dict = dict(zip(list(map(lambda x: x.replace('.bmp', ''), reso_dict['NAME'])), reso_dict['RESO']))
    
    size_list = []
    ratio_list = []
    size_mm_list = []
    for i in range(1, 6):
        anno_coco = CocoVID(os.path.join(anno_dir, str(i), 'test.json'))
        # json_file = open(os.path.join(anno_dir, str(i), 'test.json'), mode='r')
        v_ids = anno_coco.videos
        v_infos = anno_coco.load_vids(v_ids)
        for v_info in v_infos:
            v_id = v_info['id']
            reso = reso_dict[v_info['name']]
            img_ids = anno_coco.get_img_ids_from_vid(v_id)
            anno_ids = anno_coco.get_ann_ids(img_ids=img_ids, cat_ids=1)
            anno_infos = anno_coco.load_anns(anno_ids)
            for anno in anno_infos:
                size = anno['bbox'][2:]
                size_mm = size[0] * size[1] * reso **2
                size_list.append(size)
                size_mm_list.append(size_mm)
                ratio_list.append(round(size[0]/size[1], 2))
    size_list = np.array(size_list)
    size_frame = pd.DataFrame({'X': size_list[:, 0], 'Y': size_list[:, 1], 'RATIO': ratio_list, 'SIZE_MM': size_mm_list})
    def size_range(row):
        area = row['X'] * row['Y']
        if area <= 100**2:
            return '0-100'
        elif area <= 200**2:
            return '100-200'
        elif area <= 300**2:
            return '200-300'
        elif area <= 400**2:
            return '300-400'
        elif area <= 500**2:
            return '400-500'
        else:
            return '>500'

    def get_area(row):
        area = row['X'] * row['Y']
        return area


    size_frame['SIZE RANGE'] = size_frame.apply(lambda row: size_range(row), axis=1)
    size_frame['AREA'] = size_frame.apply(lambda row: get_area(row), axis=1)
    # ration_dis = sns.displot(data=size_frame, x='RATIO', kde=True)
    # ration_dis.set(title='Distribution of Nodule Ratio')
    size_dis = sns.displot(data=size_frame, x='RATIO', kde=True)
    size_dis.set(xlabel='RATIO', ylabel='Count', title='Distribution of Nodule Ratio')
    # his_plot = sns.histplot(data=size_frame, x='X',binds=10)
    # his_plot.set(xlabel='Size', ylabel='Count', title='Distribution of Nodule Size')
    size_frame.to_csv(r'/path/to/data/rootDocument/size_distribution_mm_cross_valid.csv')
    plt.show()
    print()

    
def statistic_brightness_dis():
    data_dir = r'/path/to/data/rootdata'
    dataset_dir = r'/path/to/data/rootdata/cross_valid'
    avg_list = []
    std_list = []
    
    for fold in range(1, 6):
        anno_json = json.load(open(os.path.join(dataset_dir, str(fold), 'test.json')))
        anno_brightness_dict = {}
        anno_dict = anno_json['annotations']
        image_box_dict = {}
        for bbox in anno_dict:
            image_id = bbox['image_id']
            if image_id not in image_box_dict:
                image_box_dict[image_id] = []
            image_box_dict[image_id].append(bbox)
        for image_info in tqdm(anno_json['images']):
            if image_info['id'] not in image_box_dict:
                continue
            try:
                image = Image.open(os.path.join(data_dir, image_info['file_name']))
            except:
                print(os.path.join(data_dir, image_info['file_name']))
                continue
            image = image.convert(mode='L')
            image_array = np.array(image)
            image.close()
            for bbox in image_box_dict[image_info['id']]:
                idx_list = list(map(lambda x: int(x), bbox['bbox']))
                image_patch = image_array[idx_list[1]: idx_list[1]+idx_list[3], idx_list[0]: idx_list[0]+idx_list[2]]
                # Image.fromarray(image_patch).show()
                avg_brightness = round(np.mean(image_patch), 2)
                std_brightness = round(np.std(image_patch), 2)
                anno_brightness_dict[bbox['id']] = {'AVG': avg_brightness, 'STD': std_brightness}
                avg_list.append(avg_brightness)
                std_list.append(std_brightness)
        json.dump(anno_brightness_dict, open(os.path.join(dataset_dir, str(fold), 'brightness.json'), mode='w'))
    data_df = pd.DataFrame({'AVG': avg_list, 'STD': std_list})
    data_df.to_csv(r'/path/to/data/rootDocument/brightness_distribution.csv')
    bright_dis = sns.displot(data=data_df, x='STD', kde=True)
    bright_dis.set(xlabel='Brightness STD', ylabel='Count', title='Distribution of Nodule Brightness STD')
    plt.show()

def filter_anno(value_key, value_bin, anno_dict, birghtness_dict):
    for anno in anno_dict['annotations']:
        value = birghtness_dict[anno['id']][value_key]
        if value_bin[0] <= value < value_bin[1]:
            anno['ignore'] = True
    return anno_dict

def postive_rate():
    anno_dir = r'/path/to/data/rootdata/cross_valid_final'
    save_dir = r'/save/path/'
    postive_rate_dict = {'VIDEO': [], 'POS_NUM': [], 'LEN': [], 'POS_RATE': []}
    frame_classifi_dict = {}
    for fold in range(1, 6):
        fold = str(fold)
        anno_path = os.path.join(anno_dir, fold, 'test.json')
        anno_coco = COCO(anno_path)
        anno_dict = json.load(open(anno_path))
        for video_info in anno_dict['videos']:
            positive_frames = []
            negative_frames = []
            name, v_id = video_info['name'], video_info['id']
            image_ids = [(image_info['id'], image_info['frame_id']) for image_info in anno_dict['images'] if image_info['video_id'] == v_id]
            if len(image_ids) == 0:
                print(name)
                continue
            postive_frame_num = 0
            for image_id in image_ids:
                anno_ids = anno_coco.getAnnIds(image_id[0], 1)
                if len(anno_ids) > 0:
                    positive_frames.append(image_id[1])
                    postive_frame_num += 1
                else:
                    negative_frames.append(image_id[1])
            frame_classifi_dict[name] = {'positive': positive_frames, 'negative': negative_frames}
            postive_rate_dict['VIDEO'].append(name)
            postive_rate_dict['POS_NUM'].append(postive_frame_num)
            postive_rate_dict['LEN'].append(len(image_ids))
            postive_rate_dict['POS_RATE'].append(postive_frame_num/len(image_ids))
    json.dump(frame_classifi_dict, open(os.path.join(save_dir, 'frame_classifiy_thyroid.json'), mode='w'))
    pd.DataFrame(postive_rate_dict).to_csv(os.path.join(save_dir, 'positive_rate_thyroid.csv'), index=False)

def normalize_positive_rate():
    pos_rate_df = pd.read_csv(r'/path/to/positive_rate_thyroid.csv')
    frame_classify_dict = json.load(open(r'/path/to/frame_classifiy_thyroid.json'))
    avg_postive_rate = pos_rate_df['POS_RATE'].mean()
    for idx, row in pos_rate_df.iterrows():
        if row['POS_RATE'] == 0:
            continue
        name = row['VIDEO']
        pos_rate = row['POS_RATE']
        if pos_rate > avg_postive_rate:
            sample_rate = avg_postive_rate / pos_rate
            candidate_idx = frame_classify_dict[name]['positive']
            sample_idicator = 'positive'

        else:
            sample_rate =  (1-avg_postive_rate) / (1-pos_rate)
            candidate_idx = frame_classify_dict[name]['negative']
            sample_idicator = 'negative'
        
        sampled_ids = random.sample(candidate_idx, int(len(candidate_idx) * sample_rate))
        frame_classify_dict[name]['sampled'] = sampled_ids
        frame_classify_dict[name]['indicator'] = sample_idicator
    json.dump(frame_classify_dict, open(r'/path/to/frame_sampled_thyroid.json', mode='w'))


def split_videos():
    """
    split the sequence by the positive/negative, to make every clip contains only positive/negative frames.
    """
    frame_classifiy_json = r'/path/to/frame_classifiy_thyroid.json'
    frame_classifiy_dict = json.load(open(frame_classifiy_json))
    for task_name in frame_classifiy_dict.keys():
        print(task_name)
        pos_splits = []
        pos_list = frame_classifiy_dict[task_name]['positive']
        
        neg_splits = []
        neg_list = frame_classifiy_dict[task_name]['negative']

        if len(pos_list) == 0:
            frame_classifiy_dict[task_name]['pos_splits'] = []
            frame_classifiy_dict[task_name]['neg_splits'] = [neg_list]
            continue
        
        if len(neg_list) == 0:
            frame_classifiy_dict[task_name]['neg_splits'] = []
            frame_classifiy_dict[task_name]['pos_splits'] = [pos_list]
            continue


        pos_temp = [pos_list[0]]
        neg_temp = [neg_list[0]]

        for idx in range(1, len(pos_list)):
            if (pos_list[idx] - pos_list[idx-1]) <= 1:
                pos_temp.append(pos_list[idx])
            else:
                pos_splits.append(pos_temp)
                pos_temp = [pos_list[idx]]
        pos_splits.append(pos_temp)

        for idx in range(1, len(neg_list)):
            if (neg_list[idx] - neg_list[idx-1]) <= 1:
                neg_temp.append(neg_list[idx])
            else:
                neg_splits.append(neg_temp)
                neg_temp = [neg_list[idx]]
        neg_splits.append(neg_temp)

        assert len(pos_list) == sum([len(split) for split in pos_splits])
        assert len(neg_list) == sum([len(split) for split in neg_splits])
        frame_classifiy_dict[task_name]['pos_splits'] = pos_splits
        frame_classifiy_dict[task_name]['neg_splits'] = neg_splits
    json.dump(frame_classifiy_dict, open(r'/save/path/', mode='w'))


def convert_coco_motcoco():
    root_dir = r'/path/to/data/rootdata/train_test'
    save_dir = r'/path/to/data/rootdata/train_test_cocoformat/'
    for file_name in os.listdir(root_dir):
        file_path = os.path.join(root_dir, file_name)
        origin_anno = json.load(open(file_path))
        for video_info in origin_anno['videos']:
            video_info['fps'] = 30
            video_info['width'] = 800
            video_info['height'] = 539
        for image_info in origin_anno['images']:
            image_info['mot_frame_id'] = image_info['frame_id'] + 1
        
        mot_ins_id_list = []
        mot_ins_id = 0
        for anno_info in origin_anno['annotations']:
            ins_id = str(anno_info['video_id']) + '_' + str(anno_info['instance_id'])
            if ins_id not in mot_ins_id_list:
                mot_ins_id_list.append(ins_id)
                mot_ins_id += 1
            anno_info['mot_instance_id'] = mot_ins_id
            anno_info['mot_conf'] = 1.0
            anno_info['mot_class_id'] = 1
        tar_anno = {'categories': origin_anno['categories'],
                    'images': origin_anno['images'],
                    'annotations': origin_anno['annotations'],
                    'videos': origin_anno['videos'],
                    'num_instances': len(mot_ins_id_list)}
        with open(os.path.join(save_dir, file_name), mode='w') as f:
            json.dump(tar_anno, f) 

        txt = f"[Sequence]\nname=Thyroid\nimDir={file_name}\nframeRate=30\nseqLength={len(os.listdir(os.path.join('./', file_name)))}\nimWidth=800\nimHeight539\nimExt=.tif"

    pass
                      
def get_IoU(pred_bbox, gt_bbox):
    bb1 = dict(x1=pred_bbox[0], x2=pred_bbox[2], y1=pred_bbox[1], y2=pred_bbox[3])
    bb2 = dict(x1=gt_bbox[0], x2=gt_bbox[0]+gt_bbox[2], y1=gt_bbox[1], y2=gt_bbox[1]+gt_bbox[3])
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou        

def sample_biggest_frame():
    anno_dir = r'/path/to/data/rootdataset/cross_valid'
    data_dir = r'/path/to/data/rootdata'
    save_dir = r'/path/to/data/rootSelectedForResolution'
    for fold in range(1, 6):
        fold = str(fold)
        anno_path = os.path.join(anno_dir, fold, 'test.json')
        anno_coco = COCO(anno_path)
        biggest_frame_dict = {}
        for anno_key, anno_info in anno_coco.anns.items():
            if anno_info['category_id'] == 2:
                continue
            anno_key = '_'.join([str(anno_info['video_id']), str(anno_info['instance_id'])])
            if anno_key not in biggest_frame_dict.keys():
                biggest_frame_dict[anno_key] = (0, -1)
            if anno_info['area'] > biggest_frame_dict[anno_key][0]:
                biggest_frame_dict[anno_key] = (anno_info['area'], anno_info['image_id'])
        for key in biggest_frame_dict.keys():
            image_id = biggest_frame_dict[key][1]
            image_info = anno_coco.loadImgs(image_id)
            assert len(image_info) ==  1
            image_name = image_info[0]['file_name']
            shutil.copy(os.path.join(data_dir, image_name), os.path.join(save_dir, fold+'_'+key+'.tif'))
            print(image_name)
                

def normalize_auc():
    pass


def nodule_instance_aggre():
    anno_dir = r'/path/to/data/rootdataset/train_test'
    data_dir = r'/path/to/data/rootdata'
    save_path = r'/path/to/data/rootdataset/train_test/'
    instance_dict = {'VIDEO': [], 
                     'NODULE INDEX': [],
                     'BEGIN FRAME': [],
                     'END FRAME': [],
                     'BEGIN XY': [],
                     'END XY': []}
    instance_end_dict = {}
    for mode in ['train', 'test', 'val']:
        anno_path = os.path.join(anno_dir, f'{mode}.json')
        anno_dict = json.load(open(anno_path, mode='r'))
        anno_infos = anno_dict['annotations']
        video_dict = dict(zip([video_info['id'] for video_info in anno_dict['videos']], 
                              [video_info['name'] for video_info in anno_dict['videos']]))
        frame_dict = dict(zip([frame_info['id'] for frame_info in anno_dict['images']], 
                              [frame_info for frame_info in anno_dict['images']]))
        for anno_info in anno_infos:
            if anno_info['category_id'] != 1:
                continue
            video_id = anno_info['video_id']
            video_name = video_dict[video_id]
            instance_id = anno_info['instance_id']
            v_i = video_name + '_' + str(instance_id)
            frame_id = frame_dict[anno_info['image_id']]['frame_id']
            if v_i not in instance_end_dict:
                instance_dict['VIDEO'].append(video_name)
                instance_dict['NODULE INDEX'].append(instance_id)
                instance_dict['BEGIN FRAME'].append(frame_id)
                instance_dict['END FRAME'].append(frame_id)
                instance_dict['BEGIN XY'].append(list(map(int, anno_info['bbox'])))
                instance_dict['END XY'].append(list(map(int, anno_info['bbox'])))
            instance_end_dict[v_i] = (frame_id, list(map(int, anno_info['bbox'])))
    for idx, v_name in enumerate(instance_dict['VIDEO']):
        v_id = v_name + '_' + str(instance_dict['NODULE INDEX'][idx])
        end_frame, end_bbox = instance_end_dict[v_id]
        instance_dict['END FRAME'][idx] = end_frame
        begin_frame = instance_dict['BEGIN FRAME'][idx]
        begin_bbox = instance_dict['BEGIN XY'][idx]
         
        begin_image = cv2.imread(os.path.join(data_dir, v_name, 'JPEGImages', format(begin_frame+1, '06d')+'.tif'))
        cv2.rectangle(begin_image, 
                                    (begin_bbox[0], begin_bbox[1]), 
                                    (begin_bbox[0]+begin_bbox[2], begin_bbox[1]+begin_bbox[3]),
                                    (255, 0, 0), 
                                    1)

        end_image = cv2.imread(os.path.join(data_dir, v_name, 'JPEGImages', format(end_frame+1, '06d')+'.tif'))
        cv2.rectangle(end_image, 
                                    (end_bbox[0], end_bbox[1]), 
                                    (end_bbox[0]+end_bbox[2], end_bbox[1]+end_bbox[3]),
                                    (255, 0, 0), 
                                    1)
        cv2.imwrite(os.path.join(save_path, 'instance_demo', v_id+'_begin.tif'), begin_image)
        cv2.imwrite(os.path.join(save_path, 'instance_demo', v_id+'_end.tif'), end_image)

    pd.DataFrame(instance_dict).to_csv(os.path.join(save_path, 'instance.csv'), index=False)

                
if __name__ == '__main__':
    statistic_size_dis()
