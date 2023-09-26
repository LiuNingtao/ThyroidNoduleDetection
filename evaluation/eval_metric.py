# custom/eval_metric.py
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
import numpy as np
import pandas as pd

import mmcv
from mmcv import Config, DictAction
import pickle
from mmtrack.datasets import CocoVID
from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate metric of the '
                                     'results saved in pkl format')
    parser.add_argument('--config', help='Config of the model', default=r'/path/to/config')
    parser.add_argument('--pkl-results', help='Results in pickle format', default=r'/path/to/temp_result/test.pkl')
    
    parser.add_argument('--dataset', help='the origin annotation root', default=r'/path/to/data/rootdataset/cross_valid')
    parser.add_argument('--work-root', help='the root path of work dir', default=r'/path/to/work/dir')
    # parser.add_argument('--filter-bin', help='the bin range of brightness', default=(5000, 6000))

    parser.add_argument(
        '--format-only',
        default=False,
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='bbox',
        help='Evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args()
    return args
args = parse_args()

def filter_anno(value_key, value_bin, anno_dict, birghtness_dict):
    filter_num = 0
    for anno in anno_dict:
        try:
            value = birghtness_dict[str(anno['id'])][value_key]
        except:
            print(value_key)
            value = value_bin[0] 
        if not value_bin[0] <= value < value_bin[1]:
            # anno['ignore'] = True
            anno['iscrowd'] = True
            filter_num += 1
    return anno_dict, filter_num
    

def preprocess(args, fold, task_id, bin_range_size, bin_range_brightness, video_spe=False, mm_reso=None):
    origin_result = os.path.join(args.work_root+str(fold), 'result', 'pretrain_mm_best.pkl')
    # origin_result = os.path.join(args.work_root, 'result', f'fold{str(fold)}_last.pkl')
    origin_anno = os.path.join(args.dataset, str(fold), 'test.json')
    # anno_filter_dict = json.load(open(origin_anno.replace('test.json', 'brightness.json')))
    filtered_num = 0 
    anno_coco = CocoVID(origin_anno)
    origin_anno = json.load(open(origin_anno))
    # origin_anno['annotations'], filtered_num = filter_anno('AVG', bin_range_brightness, origin_anno['annotations'], anno_filter_dict)
    for anno in origin_anno['annotations']:
        area = anno['area']
        if mm_reso:
            img_id = anno['image_id']
            img_info = anno_coco.load_imgs(ids=img_id)[0]
            video_id = img_info['video_id']
            video_name = anno_coco.load_vids(ids=video_id)[0]['name']
            reso = mm_reso[video_name]
            area = area * reso ** 2
        if not (bin_range_size[0]**2) <= anno['area'] < (bin_range_size[1]**2):
            anno['iscrowd'] = True
            filtered_num += 1
    if video_spe:
        target_video = list(filter(lambda x: x['id'] == task_id, origin_anno['videos']))
        assert len(target_video) == 1
        video_id = target_video[0]['id']
        target_video[0]['id'] = 1
        
        new_images = list(filter(lambda x: x['video_id'] == video_id, origin_anno['images']))
        new_image_ids = [x['id']-1 for x in new_images]
        ids_map = {}
        for idx in range(len(new_images)):
            old_id = new_images[idx]['id']
            new_images[idx]['id'] = idx+1
            new_images[idx]['video_id'] = 1
            ids_map[old_id] = idx+1

        new_annos = list(filter(lambda x: x['video_id'] == video_id, origin_anno['annotations']))
        for idx, anno in enumerate(new_annos):
            anno['id'] = idx + 1
            anno['image_id'] = ids_map[anno['image_id']]
            anno['video_id'] = 1

        origin_anno['images'] = new_images
        origin_anno['annotations'] = new_annos
        origin_anno['videos'] = target_video

    origin_result = np.load(origin_result, allow_pickle=True)
    if 'det_bboxes' in origin_result:
        origin_result = origin_result['det_bboxes']
    if video_spe:
        origin_result = [origin_result[i] for i in range(len(origin_result)) if i in new_image_ids]
        assert len(origin_result) == len(new_images)
    json.dump(origin_anno, open(args.pkl_results.replace('.pkl', '.json'), mode='w'))
    pickle.dump(origin_result, open(args.pkl_results, mode='wb'))
    result_len = len(list(filter(lambda x: (not x['iscrowd']) and x['category_id']==1, origin_anno['annotations'])))
    return result_len

    


def main(fold, task_id, range_bin_size, range_bin_brightness, video_spec=False):
    reso_path = r'/path/to/data/rootresolution.csv'
    reso_dict = pd.read_csv(reso_path).to_dict(orient='list')
    reso_dict = dict(zip(list(map(lambda x: x.replace('.bmp', ''), reso_dict['NAME'])), reso_dict['RESO']))
    result_len = preprocess(args, fold, task_id, range_bin_size, range_bin_brightness, video_spec, reso_dict)
    cfg = Config.fromfile(args.config)
    assert args.eval or args.format_only, (
        'Please specify at least one operation (eval/format the results) with '
        'the argument "--eval", "--format-only"')
    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.pkl_results)

    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        eval_kwargs = cfg.get('evaluation', {}).copy()
        # hard-code way to remove EvalHook args
        for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
        ]:
            eval_kwargs.pop(key, None)
        # eval_kwargs.update(dict(metric=args.eval, **kwargs))
        metrics = dataset.evaluate(outputs, **eval_kwargs)
        map_50 = metrics['bbox_mAP_50']
        # print(metrics)
        # map_50 = 0
        return map_50, result_len



if __name__ == '__main__':
    # range_list_size = [0, 100, 200, 300, 400, 500, 1000000000]
    range_list_size = [0, 1000000000]
    # range_list_brightness = [0, 20, 40, 60, 1000000]
    range_list_brightness = [0, 1000000]
    result_dict = {'SIZE': [], 'BRIGHTNESS': [], 'FOLD': [], 'MAP': [], 'RES_NUM': []}
    for i in range(len(range_list_size)):
        for j in range(len(range_list_brightness)):
            res_dict = {'FOLD': [], 'VIDEO': [], 'MAP': []}
            if i >= len(range_list_size)-1:
                bin_range_size = (range_list_size[0], range_list_size[-1])
            else:
                bin_range_size = (range_list_size[i], range_list_size[i+1])
            if j >= len(range_list_brightness)-1:
                bin_range_brightness = (range_list_brightness[0], range_list_brightness[-1])
            else:
                bin_range_brightness = (range_list_brightness[j], range_list_brightness[j+1])
            for fold in range(1, 6):
                print(str(fold)+'======')
                json_file = os.path.join(args.dataset, str(fold), 'test.json')
                videos = json.load(open(json_file))['videos']
                for video in videos:
                    map_50, result_len = main(fold, video['id'], bin_range_size, bin_range_brightness, True)
                    res_dict['FOLD'].append(fold)
                    res_dict['VIDEO'].append(video['name'])
                    res_dict['MAP'].append(map_50)
                map_50, result_len = main(fold, None, bin_range_size, bin_range_brightness, False)
                res_dict['FOLD'].append(fold)
                res_dict['VIDEO'].append('ALL')
                res_dict['MAP'].append(map_50)
                result_dict['SIZE'].append(bin_range_size)
                result_dict['BRIGHTNESS'].append(bin_range_brightness)
                result_dict['FOLD'].append(str(fold))
                result_dict['MAP'].append(map_50)
                result_dict['RES_NUM'].append(result_len)
    pd.DataFrame(res_dict).to_csv(r'/save/path', index=False)
