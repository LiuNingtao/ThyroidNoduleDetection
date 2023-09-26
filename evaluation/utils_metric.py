# custom/utils_metric.py
import numpy as np
import torch
import os
import pandas as pd
import math
from medpy.metric import dc, jc, hd, hd95, asd, assd
import json
import pickle
from collections import OrderedDict
import sys

from mmdetection.mmdet.apis import init_detector, inference_detector
from PIL import Image

def calculate_bbox_metric_pip():
    root_dir = r'/path/to/root_dir'
    mask_dir = os.path.join(root_dir, 'AutoProcessed', 'mask_png')
    save_path = os.path.join(root_dir, 'Result', 'metric_bbox_dis_no_thd.xlsx') 
    dataset_csv_path = os.path.join(root_dir, 'DataSet', 'test.csv')

    detector_cfg = r'/path/to/config'
    detector_ckpt = r'/path/to/checkpoint'
    detector = init_detector(detector_cfg, detector_ckpt, device='cuda:0')

    anno_json = json.load(open(os.path.join(root_dir, 'mmcv_dataset', 'test', 'test_tor_1.json')))
    image_bbox_dict = {}
    for i in range(len(anno_json['images'])):
        image_info = anno_json['images'][i]
        anno_info = anno_json['annotations'][i]
        if not image_info['id'] == anno_info['image_id']:
            print('ERROR'+str(image_info['id']))
        image_bbox_dict[image_info['file_name']] = anno_info['bbox']
    score_threshold = 0.5

    result_dict = {'FILE': [],
                   'CANDI':[],
                   'SCORE':[],
                   'AREA_PCTG': [],
                   'AREA_PXL': [],
                   'DIA': [],
                   'BBOX_GT_SIZE': [],
                   'BBOX_PR_SIZE': [],
                   'BBOX_DISTANCE': [],
                   'BBOX_DISTANCE_RELATIVE': [],
                   'BBOX_IoU': [],
                   'HD': [],
                   'HD95': [],
                   'ASD': [],
                   'ASSD': [],
                   'DSC': [],
                   'JAC': []
                   }
    
    file_list = pd.read_csv(dataset_csv_path).to_dict(orient='list')['ID']
    for file_name in os.listdir(os.path.join(root_dir, 'image')):
        # if os.path.exists(os.path.join(root_dir, 'image_cropped', file_name)):
        #     continue
        if (file_name.upper() not in file_list):
            continue
        print(file_name)
        image_path = os.path.join(root_dir, 'image_origin', file_name)
        mask_path = os.path.join(mask_dir, file_name)
        
        mask = np.array(Image.open(mask_path))
        area = np.sum(mask!=0)
        area_rate = area / (mask.shape[0] * mask.shape[1])
        dia = math.sqrt(area/math.pi)*2
        mask_bbox = image_bbox_dict[file_name]
        mask_center = (mask_bbox[0]+mask_bbox[2]/2, mask_bbox[1]+mask_bbox[3]/2)

        predict_result = inference_detector(detector, image_path)
        det_scores:list = predict_result[0][0][:, -1].tolist()
        if len(predict_result[1])<=0 or len(predict_result[1][0]) <= 0:
                continue
        det_scores_copy = det_scores.copy()
        det_scores.sort(reverse=True)
        for i in range(len(det_scores)):
            score = det_scores[i]
            # if score < score_threshold:
            #     continue
            current_res_idx = det_scores_copy.index(score)
            det_result = predict_result[0][0][current_res_idx].tolist()[0: -1]
            seg_result = np.array(predict_result[1][0][current_res_idx]*255, dtype=np.uint8)
            det_length = [det_result[2]-det_result[0], det_result[3]-det_result[1]]
            det_center = (det_result[0] + det_length[0]/2, det_result[1] + det_length[1]/2)
            center_distance = math.sqrt((det_center[0]-mask_center[0])**2 + (det_center[1]-mask_center[1])**2)
            center_distance_relative = center_distance / math.sqrt((mask_bbox[2]/2) ** 2 + (mask_bbox[3]/2) ** 2)
            bbox_iou = get_iou(dict(zip(['x1', 'y1', 'x2', 'y2'], det_result)), 
                            dict(zip(['x1', 'y1', 'x2', 'y2'], [mask_bbox[0], mask_bbox[1], mask_bbox[0]+mask_bbox[2], mask_bbox[1]+mask_bbox[3]])))

            result_dict['FILE'].append(file_name)
            result_dict['CANDI'].append(str(i))
            result_dict['SCORE'].append(round(score, 2))
            result_dict['AREA_PCTG'].append(round(area_rate, 2))
            result_dict['AREA_PXL'].append(area)
            result_dict['DIA'].append(round(dia, 2))
            gt_size_str = ' '.join([str(round(mask_bbox[2], 2)), str(round(mask_bbox[3], 2)), str(round(mask_bbox[2]*mask_bbox[3], 2))])
            result_dict['BBOX_GT_SIZE'].append(gt_size_str)
            pr_size_str = ' '.join([str(round(det_length[0], 2)), str(round(det_length[1], 2)), str(round(det_length[0]*det_length[1], 2))])
            result_dict['BBOX_PR_SIZE'].append(pr_size_str)
            result_dict['BBOX_DISTANCE'].append(round(center_distance, 2))
            result_dict['BBOX_DISTANCE_RELATIVE'].append(round(center_distance_relative, 2))
            result_dict['BBOX_IoU'].append(round(bbox_iou, 2))
            result_dict['HD'].append(round(hd(seg_result, mask), 2))
            result_dict['HD95'].append(round(hd95(seg_result, mask), 2))
            result_dict['ASD'].append(round(asd(seg_result, mask), 2))
            result_dict['ASSD'].append(round(assd(seg_result, mask), 2))
            result_dict['DSC'].append(round(dc(seg_result, mask), 2))
            result_dict['JAC'].append(round(jc(seg_result, mask), 2))
    
    pd.DataFrame(result_dict).to_excel(save_path, index=False)
