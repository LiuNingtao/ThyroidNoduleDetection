# custom/metric.py
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from matplotlib import pyplot as plt
import math
from pycocotools.coco import COCO
from scipy.special import rel_entr, kl_div
from PIL import Image
from tqdm import tqdm
import seaborn as sns
from mmtrack.datasets import CocoVID

reso_path = r'/path/to/data/rootresolution.csv'
reso_dict = pd.read_csv(reso_path).to_dict(orient='list')
reso_dict = dict(zip(list(map(lambda x: x.replace('.bmp', ''), reso_dict['NAME'])), reso_dict['RESO']))


def load_pkl():
    path = r'/path/to/pkl'
    array = np.load(path, allow_pickle=True)
    print(array)

def get_metric_tracking(hit_num_ths = 1, score_ths = 0.8, iou_ths = 0.5, dis_ths=1):
    anno_dir = r'/path/to/coco_dataset'
    result_dir = r'/path/to/result'
    save_dir = r'/save/path'


    result_dict = {'FOLD': [], 'TP': [], 'FP': [], 'FN': [], 'TN': []}
    for fold in range(0, 1):
        anno_file = open(os.path.join(anno_dir, 'train_test', 'test_cocoformat.json'), mode='r', encoding='utf8')
        anno = json.load(anno_file)

        result_file = os.path.join(result_dir, f'test.pkl')
        result = np.load(result_file, allow_pickle=True)
        result = result['track_bboxes']

        image_bbox_dict = {}
        instance_bbox_dict = {}
        instance_bbox_pred_dict = {}
        for anno_info in anno['annotations']:
            image_id = anno_info['image_id']
            instance_id = anno_info['instance_id']
            if image_id not in image_bbox_dict:
                image_bbox_dict[image_id] = []
            image_bbox_dict[image_id].append(anno_info)
            if instance_id not in instance_bbox_dict:
                instance_bbox_dict[instance_id] = []
            instance_bbox_dict[instance_id].append(anno_info)
        
        for idx, track_bbox in enumerate(result):
            nodule_bbox_list = track_bbox[0]
            if nodule_bbox_list.shape[0] <=0:
                continue
            for bbox in nodule_bbox_list:
                instance_id_pred = bbox[0]
                instance_score = bbox[-1]
                if instance_score < score_ths:
                    continue
                if instance_id_pred not in instance_bbox_pred_dict:
                    instance_bbox_pred_dict[instance_id_pred] = []
                instance_bbox_pred_dict[instance_id_pred].append({'score': instance_score, 
                                                                  'ins_id': instance_id_pred, 
                                                                  'image_id': idx+1, 
                                                                  'bbox': bbox[1: 5]})
        
        gt_hit_dict, pred_hit_dict = cal_TP(instance_bbox_pred_dict, instance_bbox_dict, iou_ths, dis_ths)
        gt_hitted_num = 0
        pred_hitted_num = 0
        for key in gt_hit_dict:
            if gt_hit_dict[key] >= hit_num_ths:
                gt_hitted_num += 1
        for key in pred_hit_dict:
            if pred_hit_dict[key] >= hit_num_ths:
                pred_hitted_num += 1
        result_dict['FOLD'].append(fold)
        result_dict['TP'].append(gt_hitted_num)
        result_dict['FN'].append(len(list(instance_bbox_dict.keys())) - gt_hitted_num)
        result_dict['FP'].append(len(list(instance_bbox_pred_dict.keys())) - pred_hitted_num)
        result_dict['TN'].append(1)
    result_dict['FOLD'].append('ALL')
    result_dict['TP'].append(sum(result_dict['TP']))
    result_dict['FN'].append(sum(result_dict['FN']))
    result_dict['FP'].append(sum(result_dict['FP']))
    result_dict['TN'].append(sum(result_dict['TN']))
    pd.DataFrame(result_dict).to_csv(os.path.join(save_dir, f'confusion_metrix_last_hit_{hit_num_ths}_socre_{score_ths}_iou_{iou_ths}_dis_{dis_ths}.csv'), index=False)
    return result_dict

def get_dis(pred_bbox, gt_bbox):
    bb1 = dict(x1=pred_bbox[0], x2=pred_bbox[2], y1=pred_bbox[1], y2=pred_bbox[3])
    bb2 = dict(x1=gt_bbox[0], x2=gt_bbox[0]+gt_bbox[2], y1=gt_bbox[1], y2=gt_bbox[1]+gt_bbox[3])
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_mid_1, y_mid_1 = (bb1['x1'] + bb1['x2']) / 2, (bb1['y1'] + bb1['y2']) / 2
    w1, h1 = bb1['x2'] - bb1['x1'], bb1['y2'] + bb1['y1']

    x_mid_2, y_mid_2 = (bb2['x1'] + bb2['x2']) / 2, (bb2['y1'] + bb2['y2']) / 2
    w2, h2 = bb2['x2'] - bb2['x1'], bb2['y2'] + bb2['y1']

    dis = math.sqrt((x_mid_2 - x_mid_1) ** 2 + (y_mid_2 - y_mid_1) ** 2)
    die1, die2 = math.sqrt(w1 ** 2 + h1 ** 2)/2, math.sqrt(w2 ** 2 + h2 ** 2)/2
    fract = dis / (die2 / 2)
    return dis, fract


def get_AUC():
    IoU_score = 0.5
    dis_ths = 0.5
    tpr_list = []
    fpr_list = []
    for score_ths in np.arange(0, 1, 0.1):
        print(score_ths)
        result = get_metric_tracking(score_ths=score_ths, iou_ths=IoU_score, dis_ths=dis_ths)
        tp, fn, fp, tn = result['TP'][-1], result['FN'][-1], result['FP'][-1], result['TN'][-1]
        tpr_list.append(tp/(tp+fn))
        fpr_list.append(fp/(tn+fp))



def cal_TP(instance_bbox_pred_dict, instance_bbox_dict, iou_ths=0.5, dis_ths=1):
    gt_hit_dict = {}
    pred_hit_dict = {}
    for ins_id_pred in instance_bbox_pred_dict:
        for ins_id_gt in instance_bbox_dict:
            pred_list = instance_bbox_pred_dict[ins_id_pred]
            gt_list = instance_bbox_dict[ins_id_gt]
            
            image_id_inject = list(set([anno['image_id'] for anno in gt_list]) & set([pred['image_id'] for pred in pred_list]))
            for image_id in image_id_inject:
                frame_pred = list(filter(lambda x: x['image_id'] == image_id, pred_list))
                frame_gt = list(filter(lambda x: x['image_id'] == image_id, gt_list))
                assert len(frame_pred) == 1 and len(frame_gt) == 1, print('More than 1 bbox')
                bbox_pred, bbox_gt = frame_pred[0]['bbox'], frame_gt[0]['bbox']
                dis, frac = get_dis(bbox_pred, bbox_gt)
                iou = get_IoU(bbox_pred, bbox_gt)
                if frac <= dis_ths or iou >= iou_ths:
                    if ins_id_gt not in gt_hit_dict:
                        gt_hit_dict[ins_id_gt] = 0
                    gt_hit_dict[ins_id_gt] += 1

                    if ins_id_pred not in pred_hit_dict:
                        pred_hit_dict[ins_id_pred] = 0
                    pred_hit_dict[ins_id_pred] += 1
    return gt_hit_dict, pred_hit_dict
        

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


def confusion_metrix_dataset():
    fold_num = 5
    result_dir = '/path/to/result'
    anno_dir = '/path/to/annotation'
    imgae_sampled_json = json.load(open(r'/path/to/frame_sampled.json'))
    save_path = '/save/path'
    metric_dict = {'ID': [], 
                   'GT': [],
                   'NAME': [], 
                   'SCORE': [], 
                   'POS_RATE': [], 
                   'SCORE_FIXED': []}
    for fold in range(fold_num):
        fold = str(fold+1)
        result_path = os.path.join(result_dir.format(fold), 'pretrain.pkl')
        predict_result = np.load(result_path, allow_pickle=True)
        predict_result = predict_result['det_bboxes']
        anno_path = os.path.join(anno_dir.format(fold), 'test.json')
        anno_info = json.load(open(anno_path, mode='r'))
        video_anno = anno_info['videos']
        image_anno = anno_info['images']
        bbox_anno = anno_info['annotations']
        for video in video_anno:
            video_id = video['id']
            video_name = video['name']
            print(video_name)
            images = [image_info for image_info in image_anno if image_info['video_id'] == video_id]
            if 'indicator' in imgae_sampled_json[video_name]:
                if imgae_sampled_json[video_name]['indicator'] == 'positive':
                    sampled_frame_ids = imgae_sampled_json[video_name]['sampled'] + imgae_sampled_json[video_name]['negative']
                else:
                    sampled_frame_ids = imgae_sampled_json[video_name]['sampled'] + imgae_sampled_json[video_name]['positive']
                assert len(sampled_frame_ids) == len(set(sampled_frame_ids))
                images = [image_info for image_info in images if image_info['frame_id'] in sampled_frame_ids]
                assert len(images) == len(sampled_frame_ids)
            pos_img_len = 0
            score_list = []
            for image in images:
                bbox_list = [bbox_info for bbox_info in bbox_anno if bbox_info['image_id'] == image['id'] and bbox_info['category_id'] == 1]
                if len(bbox_list) > 0:
                    pos_img_len += 1
                if len(predict_result[image['id']-1][0])>0:
                    score_list.append(predict_result[image['id']-1][0][0][-1])
                else:
                    score_list.append(0)
            pos_rate = pos_img_len / len(images)
            avg_score = np.mean(score_list)
            if pos_img_len > 0:
                avg_score_fixed = avg_score + (1-pos_rate)
                avg_score_fixed = min(1, avg_score_fixed)
                metric_dict['GT'].append(1)
            else:
                avg_score_fixed = avg_score
                metric_dict['GT'].append(0)

            
            metric_dict['ID'].append(video_id)
            metric_dict['NAME'].append(video['name'])
            metric_dict['POS_RATE'].append(pos_rate)
            metric_dict['SCORE'].append(avg_score)
            metric_dict['SCORE_FIXED'].append(avg_score_fixed)
    pd.DataFrame(metric_dict).to_csv(os.path.join(save_path, 'video_metric.csv'), index=False)
    auc_socre = roc_auc_score(y_true= metric_dict['GT'], y_score=metric_dict['SCORE'])
    fpr, tpr, _ = roc_curve(y_true= metric_dict['GT'], y_score=metric_dict['SCORE'])
    draw_auc_curve(fpr=fpr, tpr=tpr, auc=auc_socre)
    print(auc_socre)

def auc_video_split():
    fold_num = 5
    result_dir = '/path/to/result'
    anno_dir = '/path/to/data/rootdataset/cross_valid/{}'
    imgae_split_json = json.load(open(r'/path/to/frame_splits_thyroid.json'))
    save_path = '/save/path'
    metric_dict = {'ID': [], 
                   'GT': [],
                   'NAME': [], 
                   'SCORE': []}
    for fold in range(fold_num):
        fold = str(fold+1)
        result_path = os.path.join(result_dir.format(fold), 'pretrain_mm_best.pkl')
        predict_result = np.load(result_path, allow_pickle=True)
        predict_result = predict_result['det_bboxes']
        anno_path = os.path.join(anno_dir.format(fold), 'test.json')
        anno_info = json.load(open(anno_path, mode='r'))
        video_anno = anno_info['videos']
        image_anno = anno_info['images']
        bbox_anno = anno_info['annotations']
        for video in video_anno:
            video_id = video['id']
            video_name = video['name']
            print(video_name)
            frames_video = [image_info for image_info in image_anno if image_info['video_id'] == video_id]
            video_id = str(video_id)
            for idx, pos_split in enumerate(imgae_split_json[video_name]['pos_splits']):
                images = [image_info for image_info in frames_video if image_info['frame_id'] in pos_split]
                if len(images) != len(pos_split):
                    print(f'images: {str(len(images))}, splits: {str(len(pos_split))}')
                assert len(images) == len(pos_split), print(f'images: {str(len(images))}, splits: {str(len(pos_split))}')
                if len(images) <= 5:
                    continue
                avg_score = get_avg_score(images, predict_result)
                metric_dict['GT'].append(1)
                metric_dict['ID'].append(video_id+'_POS_'+str(idx))
                metric_dict['NAME'].append(video['name'])
                metric_dict['SCORE'].append(avg_score)

            
            for idx, neg_split in enumerate(imgae_split_json[video_name]['neg_splits']):
                images = [image_info for image_info in frames_video if image_info['frame_id'] in neg_split]
                assert len(images) == len(neg_split)
                if len(images) <= 5:
                    continue
                avg_score = get_avg_score(images, predict_result)
                metric_dict['GT'].append(0)
                metric_dict['ID'].append(video_id+'_NEG_'+str(idx))
                metric_dict['NAME'].append(video['name'])
                metric_dict['SCORE'].append(avg_score)
     
    pd.DataFrame(metric_dict).to_csv(os.path.join(save_path, 'video_split_metric_pretrain.csv'), index=False)
    auc_socre = roc_auc_score(y_true= metric_dict['GT'], y_score=metric_dict['SCORE'])
    fpr, tpr, _ = roc_curve(y_true= metric_dict['GT'], y_score=metric_dict['SCORE'])
    draw_auc_curve(fpr=fpr, tpr=tpr, auc=auc_socre)
    print(auc_socre)

def get_avg_score(images, predict_result):
    score_list = []
    for image in images:
        if len(predict_result[image['id']-1][0])>0:
            score_list.append(predict_result[image['id']-1][0][0][-1])
        else:
            score_list.append(0)
    avg_score = np.mean(score_list)
    return avg_score

def draw_auc_curve(fpr, tpr, auc):
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC for video scale normailzed")
    plt.legend(loc="lower right")
    plt.show()
    print('')

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def get_KL_distance():
    is_ordered = True
    for fold in range(1, 6):
        fold = str(fold)
        anno_path = f'/path/to/data/annotation'
        result_path = f'/path/to/data/result'
        save_path = r'/save/path/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        anno_dict = json.load(open(anno_path))
        anno_coco = COCO(anno_path)
        result_list = np.load(result_path, allow_pickle=True)['det_bboxes']
        # result_list = np.load(result_path, allow_pickle=True)
        result_dict = {'video': [], 'kl_dis': []}
        for video_info in anno_dict['videos']:
            score_distrib = []
            gt_distrib = []
            video_id = video_info['id']
            image_idx_list = [image_info['id'] for image_info in anno_dict['images'] if image_info['video_id'] == video_id]
            for image_idx in image_idx_list:
                anno_ids = anno_coco.getAnnIds(imgIds=image_idx, catIds=1)
                if len(anno_ids) > 0:
                    gt_distrib.append(1)
                else:
                    gt_distrib.append(0)
                image_res = result_list[image_idx - 1]
                if image_res[0].shape[0] == 0:
                    score_distrib.append(0)
                else:
                    score_distrib.append(image_res[0][0][-1])
            score_distrib, gt_distrib = np.array(score_distrib), np.array(gt_distrib)
            # score_distrib = score_distrib / np.sum(score_distrib)
            # gt_distrib = gt_distrib / np.sum(gt_distrib)
            KL_dis = kl_divergence(gt_distrib, score_distrib)
            result_dict['video'].append(video_info['name'])
            result_dict['kl_dis'].append(KL_dis)
        pd.DataFrame(result_dict).to_csv(os.path.join(save_path, f'fold_{fold}.csv'), index=False)

def statistic_contrast():
    anno_dir = '/path/to/anno/'
    result_dir = '/path/to/result'
    data_dir = r'/path/to/data/rootdata'
    contrast_dict = {'BRIGHTNESS_N': [], 'SIZE': [], 'BRIGHTNESS_T_1': [], 'BRIGHTNESS_T_2': [], 'TP': []}
    def is_inner_thyroid(nodule_bbox, thyroid_bbox):
        if (thyroid_bbox[0] <= nodule_bbox[0]) and (thyroid_bbox[1] <= nodule_bbox[1]) and \
            ((thyroid_bbox[0] + thyroid_bbox[2]) >= (nodule_bbox[0] + nodule_bbox[2])) and ((thyroid_bbox[1] + thyroid_bbox[3]) >= (nodule_bbox[1] + nodule_bbox[3])):
            return True
        return False

    def is_TP(result_list, anno_bbox, prob_thres=0.5, iou_thres=0.4):
        result_list = [r for r in result_list if r[-1] >= prob_thres]
        for res_bbox in result_list:
            if get_IoU(res_bbox[0: 4], anno_bbox) >= iou_thres:
                return True
        return False

    instance_buffered_dict = {}
    for fold in range(1, 6):
        fold = str(fold)
        anno_path = os.path.join(anno_dir.format(fold), 'test.json')
        anno_coco = CocoVID(anno_path)
        result = np.load(result_dir.format(fold), allow_pickle=True)['det_bboxes']
        assert len(result) == len(anno_coco.imgs)

        for image_key, image_info in tqdm(anno_coco.imgs.items()):
            image_origin = Image.open(os.path.join(data_dir, image_info['file_name']))
            image = np.array(image_origin)
            anno_idx_noudle = anno_coco.getAnnIds(image_key, [1])
            anno_idx_thyroid = anno_coco.getAnnIds(image_key, [2])
            v_name = anno_coco.videos[image_info['video_id']]['name']
            reso = reso_dict[v_name]
            for nodule_idx in anno_idx_noudle:
                nodule_info  = anno_coco.loadAnns(nodule_idx)[0]
                nodule_bbox = nodule_info['bbox']
                if is_TP(result_list=result[image_key-1][0], anno_bbox=nodule_bbox):
                    tp = 'TP'
                else:
                    tp = 'FN'
                nodule_bbox = [int(b) for b in nodule_bbox]
                thyroid_out = [nodule_bbox[0]-20, nodule_bbox[1]-20, nodule_bbox[2]+40, nodule_bbox[3]+40]
                for thyroid_idx in anno_idx_thyroid:
                    thyroid_info = anno_coco.loadAnns(thyroid_idx)[0]
                    thyroid_bbox = thyroid_info['bbox']
                    thyroid_bbox = [int(b) for b in thyroid_bbox]
                    if is_inner_thyroid(nodule_bbox, thyroid_bbox):
                        thyroid_out = thyroid_bbox
                        break
                nodule_avg = np.mean(image[int(nodule_bbox[1]+0.3*nodule_bbox[3]): int(nodule_bbox[1]+0.7*nodule_bbox[3]), int(nodule_bbox[0]+0.3*nodule_bbox[2]): int(nodule_bbox[0]+0.7*nodule_bbox[2])])
                thyroid_avg = np.mean(image[thyroid_out[1]: thyroid_out[1]+thyroid_out[3], thyroid_out[0]: thyroid_out[0]+thyroid_out[2]])
                thyroid_mask = np.ones_like(image)
                thyroid_mask[thyroid_out[1]: thyroid_out[1]+thyroid_out[3], thyroid_out[0]:thyroid_out[0]+thyroid_out[2]] = 0
                thyroid_mask[nodule_bbox[1]: nodule_bbox[1]+nodule_bbox[3], nodule_bbox[0]:nodule_bbox[0]+nodule_bbox[2]] = 1
                # image_origin.show()
                # Image.fromarray(np.array(thyroid_mask*255, dtype=np.uint8)).show()
                thyroid_mask = np.array(thyroid_mask, dtype=bool)
                thyroid_mask = np.ma.array(image, mask=thyroid_mask)

                contrast_dict['TP'].append(tp)
                contrast_dict['BRIGHTNESS_N'].append(nodule_avg)
                contrast_dict['SIZE'].append(np.mean(nodule_info['area'] * (reso **2)))
                contrast_dict['BRIGHTNESS_T_1'].append(thyroid_avg)
                contrast_dict['BRIGHTNESS_T_2'].append(thyroid_mask.mean())
                
                key_instance = fold+'_'+str(nodule_info['video_id'])+'_'+str(nodule_info['instance_id'])
                if key_instance not in instance_buffered_dict:
                    instance_buffered_dict[key_instance] = {'TP': [], 'BRIGHTNESS_N': [], 'SIZE': [], 'BRIGHTNESS_T_1': [], 'BRIGHTNESS_T_2': [], 'VIDEO': []}
                instance_buffered_dict[key_instance]['TP'].append(tp)
                instance_buffered_dict[key_instance]['BRIGHTNESS_N'].append(nodule_avg)
                instance_buffered_dict[key_instance]['SIZE'].append(nodule_info['area']* (reso **2))
                instance_buffered_dict[key_instance]['BRIGHTNESS_T_1'].append(thyroid_avg)
                instance_buffered_dict[key_instance]['BRIGHTNESS_T_2'].append(thyroid_mask.mean())
                instance_buffered_dict[key_instance]['VIDEO'].append(v_name)
    data_frame = pd.DataFrame(contrast_dict)
    data_frame['CONTRAST_1'] = data_frame['BRIGHTNESS_N']/data_frame['BRIGHTNESS_T_1'] - 1
    # data_frame['CONTRAST_1'] = data_frame['CONTRAST_1'].abs()
    data_frame['CONTRAST_2'] = data_frame['BRIGHTNESS_N']/data_frame['BRIGHTNESS_T_2'] - 1
    # data_frame['CONTRAST_2'] = data_frame['CONTRAST_2'].abs()
    # sns.scatterplot(data=data_frame, x='BRIGHTNESS_N', y='CONTRAST_2', hue='TP')
    # plt.show()
    data_frame.to_excel(r'/save/path', index=False)

    instance_dict = {'INSTANCE_ID':[], 'VIDEO': [], 'BRIGHTNESS_N': [], 'SIZE': [], 'BRIGHTNESS_T_1': [], 'BRIGHTNESS_T_2': [], 'TP': []}
    for key in instance_buffered_dict.keys():
        instance_dict['INSTANCE_ID'].append(key)
        instance_dict['VIDEO'].append(instance_buffered_dict[key]['VIDEO'][0])
        instance_dict['BRIGHTNESS_N'].append(np.mean(instance_buffered_dict[key]['BRIGHTNESS_N']))
        instance_dict['SIZE'].append(np.mean(instance_buffered_dict[key]['SIZE']))
        instance_dict['BRIGHTNESS_T_1'].append(np.mean(instance_buffered_dict[key]['BRIGHTNESS_T_1']))
        instance_dict['BRIGHTNESS_T_2'].append(np.mean(instance_buffered_dict[key]['BRIGHTNESS_T_2']))
        instance_dict['TP'].append('TP' if 'TP' in instance_buffered_dict[key]['TP'] else 'FN')
    data_frame = pd.DataFrame(instance_dict)
    data_frame['CONTRAST_1'] = data_frame['BRIGHTNESS_N']/data_frame['BRIGHTNESS_T_1'] - 1
    # data_frame['CONTRAST_1'] = data_frame['CONTRAST_1'].abs()
    data_frame['CONTRAST_2'] = data_frame['BRIGHTNESS_N']/data_frame['BRIGHTNESS_T_2'] - 1
    # data_frame['CONTRAST_2'] = data_frame['CONTRAST_2'].abs()
    # sns.scatterplot(data=data_frame, x='BRIGHTNESS_N', y='CONTRAST_2', hue='TP')
    # plt.show()
    data_frame.to_excel(r'/save/path', index=False)

def nodule_TPR_with_thres():
    """
    calculate the TPR of nodules with only the nodule bigger than the size threshold be counted.
    """
    result_path = r'/path/to/result'
    save_path = r'/save/path'
    result_df = pd.read_excel(result_path)
    TPR_dict = {'THRES': [], 'TPR': []}
    for thres in range(0, 1200, 10):
        result_df = result_df[result_df['SIZE_MM']>=thres]
        TPR = len(result_df[result_df['TP']=='TP']) / len(result_df)
        TPR_dict['THRES'].append(thres)
        TPR_dict['TPR'].append(round(TPR, 2))
    pd.DataFrame(TPR_dict).to_csv(save_path, index=False)
    



if __name__ == '__main__':
    # get_metric_tracking()
    # get_AUC()
    # confusion_metrix_dataset()
    # auc_video_split()
    get_KL_distance()
    # statistic_contrast()
    # nodule_TPR_with_thres()