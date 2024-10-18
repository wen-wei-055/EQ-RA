import argparse
import os
import numpy as np
import json
import pickle
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
import h5py
from scipy.stats import norm
from tqdm import tqdm
import pandas as pd
import torch

import models
from models import EnsembleEvaluateModel
import loader
import util_origin
#import plots

from util_origin import generator_from_config

EARTH_RADIUS = 6371

sns.set(font_scale=1.5)
sns.set_style('ticks')


def reassigned_label(st_pga_times, st_coords, survival_list):  # (stations, 7) , (stations,3)
    final_st_pga_times = np.zeros((len(stations_table.keys()), st_pga_times.shape[1])) - 999
    for st_pga_time, st_coord in zip(st_pga_times, st_coords):
        station_key = f"{st_coord[0]},{st_coord[1]},{st_coord[2]}"
        position = stations_table[station_key]  # 0~248
        final_st_pga_times[position] = st_pga_time
        
    # 保留原始內容在tmp_st_pgas_times，要tmp_st_pgas_times找哪些地方是-999
    tmp_st_pgas_times = final_st_pga_times.copy()   
    final_st_pga_times[final_st_pga_times==-999] = np.nan  
    
    for i, tmp_st_pga in enumerate(tmp_st_pgas_times):
        if np.sum(tmp_st_pga == -999) > 0: #代表有-999的, 需要替補label
            topK_nearest = nearest_topK_stations[str(i)]
            topK_nearest_pga = final_st_pga_times[topK_nearest]
            final_st_pga_times[i] = np.nanmean(topK_nearest_pga, axis=0)
            
    # 過濾未使用測站
    final_st_pga_times[survival_list==0] = np.nan
    return final_st_pga_times
    

def calculate_warning_times(config, model_list, data, event_metadata, batch_size, sampling_rate=100,
                            times=np.arange(0.5, 25, 0.8), alpha=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), use_multiprocessing=False,
                            dataset_id=None, device='cuda'):
    training_params = config['training_params']

    if dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[dataset_id]
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]

    n_pga_targets = config['model_params'].get('n_pga_targets', 0)

    generator_params['magnitude_resampling'] = 1
    generator_params['batch_size'] = batch_size
    generator_params['transform_target_only'] = generator_params.get('transform_target_only', True)
    generator_params['upsample_high_station_events'] = None

    alpha = np.array(alpha)
    
    f = h5py.File(data_path[0], 'r')
    g_data = f['data']
    thresholds = f['metadata']['pga_thresholds'][()]
    time_before = f['metadata']['time_before'][()]

    if generator_params.get('coord_keys', None) is not None:
        raise NotImplementedError('Fixed coordinate keys are not implemented in location evaluation')

    if 'data_file' in event_metadata.columns:
        event_key = 'data_file'
    else:
        event_key = '#EventID'

    full_predictions = []
    no_survival_full_predictions = []
    coord_keys = util_origin.detect_location_keys(event_metadata.columns)

    max_stations = config['model_params'].get('max_stations')
    latlon_IDtable = latlondep_ID(stations_table)

    stations_channel_boolean = [1 for _ in list(stations_table.keys())]
    stations_channel_class = [1 for _ in list(stations_table.keys())]


    for i, _ in tqdm(enumerate(event_metadata.iterrows()), total=len(event_metadata)):
        event = event_metadata.iloc[i]
        event_metadata_tmp = event_metadata.iloc[i:i+1]
        data_tmp = {key: val[i:i+1] for key, val in data.items()}
        generator_params['translate'] = False
        n_pga_targets = 249
        generator = util_origin.PreloadedEventGenerator(data=data_tmp,
                                                 stations_table=stations_table,
                                                 event_metadata=event_metadata_tmp,
                                                 coords_target=True,
                                                 cutout=(0, 3000),
                                                 pga_targets=n_pga_targets,
                                                 max_stations=max_stations,
                                                 sampling_rate=sampling_rate,
                                                 select_first=True,
                                                 shuffle=False,
                                                 pga_mode=True,
                                                 stations_channel_boolean=stations_channel_boolean,
                                                 stations_channel_class=stations_channel_class,
                                                 latlon_IDtable=latlon_IDtable,
                                                 **generator_params)

        cutout_generator = util_origin.CutoutGenerator(generator, times, sampling_rate=sampling_rate)

        #(123, 20, 50, 3): (時間個數(0.5~25, 每0.2一個time),  測站數, 5 * 10個ensemble , 3)
        # Assume PGA output at index 2
        if use_multiprocessing:
            workers = 0
            
        prediction, survival_list = model_list.predict_generator(cutout_generator, workers=0, use_multiprocessing=False) 
    
        prediction = prediction.reshape((len(times), -1) + prediction.shape[2:])
        
        pga_pred = torch.Tensor(prediction)
        
        pga_times_pre = np.zeros((pga_pred.shape[1], thresholds.shape[0], alpha.shape[0]), dtype=int)

        alpha = torch.Tensor(alpha)
        for j, level in enumerate(np.log10(thresholds*10)):
            
            prob = torch.sum(
                pga_pred[:, :, :, 0] * (1 - norm.cdf((level - pga_pred[:, :, :, 1]) / pga_pred[:, :, :, 2])),
                dim=-1)
            
            prob = torch.unsqueeze(prob, -1)
            
            exceedance = torch.gt(prob,alpha)  # Shape: times, stations, 1
            exceedance = np.pad(exceedance, ((1, 0), (0, 0), (0, 0)), mode='constant')
            pga_times_pre[:, j] = np.argmax(exceedance, axis=0)

        pga_times_pre -= 1
        pga_times_pred = np.zeros_like(pga_times_pre, dtype=float)
        pga_times_pred[pga_times_pre == -1] = np.nan
        pga_times_pred[pga_times_pre > -1] = times[pga_times_pre[pga_times_pre > -1]]  # 每0.2秒進入模型一次，如果有超過pga閾值就放入pga_times_pred

        g_event = g_data[str(event[event_key])]
        pga_times_true_pre = g_event['pga_times'][()]
        coords = g_event['coords'][()]
        channels = g_event['trace_filename'][()]
        coords_event = event[coord_keys]
        
        # label重製
        pga_times_true = reassigned_label(pga_times_true_pre, coords, survival_list)

        # 新增事件最早pick
        pick_offset = np.min(g_event['p_picks'][()])
        
        pga_times_true = pga_times_true[filter_borehole]
        pga_times_true[pga_times_true == 0] = np.nan
        pga_times_true[pga_times_true != 0] = (pga_times_true[pga_times_true != 0]) / sampling_rate - time_before  #在座資料集的時候有 + time_before
        
        coords = (np.array([[float(x) for x in row] for row in [x.split(',')[:-1] for x in stations_table]]))[-249:]
        coords = coords[filter_borehole]
        dist = np.zeros(coords.shape[0])
        for j, station_coords in enumerate(coords):
            dist[j] = geodesic(station_coords[:2], coords_event[:2]).km
        dist = np.sqrt(dist ** 2 + coords_event[2] ** 2)
        
        full_predictions += [(pga_times_pred, pga_times_true, dist)]

    return full_predictions
    
def latlondep_ID(stations_table):
    appear_first = {}
    appear_dic = {}
    i = 0
    for id_name, instrument_name in enumerate(stations_table):
        latlondep = ','.join(instrument_name.split(',')[:3])
        if latlondep not in appear_first.keys():
            appear_first[latlondep] = i
            appear_dic[id_name] = i
            i += 1
        if latlondep in appear_first.keys():
            appear_dic[id_name] = appear_first[latlondep]
    return appear_dic

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True)
    parser.add_argument('--weight_file', type=str)  # If unset use latest model
    parser.add_argument('--times', type=str, default='0.5,1,2,4,8,16,25')  # Has only performance implications
    parser.add_argument('--max_stations', type=int)  # Overwrite max stations value from config
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val', action='store_true')  # Evaluate on val set
    parser.add_argument('--n_pga_targets', type=int)  # Overwrite number of PGA targets
    parser.add_argument('--head_times', type=str, default=True)  # Evaluate warning times
    parser.add_argument('--blind_time', type=float, default=0.5)  # Time of first evaluation after first P arrival
    parser.add_argument('--alpha', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')  # 機率大於alpha以上就會發布警報 Probability thresholds alpha
    parser.add_argument('--additional_data', type=str)  # Additional data set to use for evaluation
    parser.add_argument('--dataset_id', type=int)  # ID of dataset to evaluate on, in case of joint training
    parser.add_argument('--wait_file', type=str)  # Wait for this file to exist before starting evaluation
    parser.add_argument('--ensemble_member', action='store_true')  # Task to evaluate is an ensemble member
                                                                   # (not the full ensembel)
    parser.add_argument('--loss_limit', type=float) # In ensemble model, discard members with loss above this limit
    parser.add_argument('--no_multiprocessing', action='store_true')
    args = parser.parse_args()
    
    torch.set_num_threads(5)

    if args.wait_file is not None:
        util_origin.wait_for_file(args.wait_file)

    times = [float(x) for x in args.times.split(',')]

    config = json.load(open(os.path.join(args.experiment_path, 'config.json'), 'r'))
    training_params = config['training_params']

    stations_table = json.load(open(training_params['station_json_file'], 'r'))
    max_stations = len(stations_table)

    # 過濾井下海底測站
    filter_borehole = json.load(open('dataset_configs/idx_st_loc_filter.json'))
    filter_borehole = [int(key) for key, value in filter_borehole.items() if value == 1]
        
    nearest_topK_stations = json.load(open(training_params['topK_nearest_file'],'r'))

    device = torch.device(training_params['device'] if torch.cuda.is_available() else "cpu")

    
    if (args.dataset_id is None) and (isinstance(training_params['data_path'], list) and
                                      len(training_params['data_path']) > 1):
        raise ValueError('dataset_id needs to be set for experiments with multiple input data sets.')
    if (args.dataset_id is not None) and not (isinstance(training_params['data_path'], list) and
                                              len(training_params['data_path']) > 1):
        raise ValueError('dataset_id may only be set for experiments with multiple input data sets.')

    if args.dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[args.dataset_id]
        n_datasets = len(training_params['data_path'])
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]
        n_datasets = 1
        
    batch_size = generator_params['batch_size']
    key = generator_params.get('key', 'MA')
    pos_offset = generator_params.get('pos_offset', (-21, -69))
    pga_key = generator_params.get('pga_key', 'pga')

    if args.blind_time != 0.5:
        suffix = f'_blind{args.blind_time:.1f}'
    else:
        suffix = ''

    if args.dataset_id is not None:
        suffix += f'_{args.dataset_id}'

    if args.val:
        output_dir = os.path.join(args.experiment_path, f'evaluation{suffix}', 'val')
        data_path = training_params['val_data_path']
        test_set = False
    else:
        output_dir = os.path.join(args.experiment_path, f'evaluation{suffix}', 'test')
        data_path = training_params['test_data_path']
        test_set = True

    if not os.path.isdir(os.path.join(args.experiment_path, f'evaluation{suffix}')):
        os.mkdir(os.path.join(args.experiment_path, f'evaluation{suffix}'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(training_params['station_first_appearance'],'r') as f: first_appearance_list = json.load(f)
    with open(training_params['station_last_appearance'],'r') as f: last_appearance_list = json.load(f)
    first_appearance_list = list(first_appearance_list.values())
    last_appearance_list = list(last_appearance_list.values())

    shuffle_train_dev = generator_params.get('shuffle_train_dev', False)
    custom_split = generator_params.get('custom_split', None)
    overwrite_sampling_rate = training_params.get('overwrite_sampling_rate', None)
    min_mag = generator_params.get('min_mag', None)
    mag_key = generator_params.get('key', 'MA')
    event_metadata, data, metadata = loader.load_events(data_path,
                                                        shuffle_train_dev=shuffle_train_dev,
                                                        custom_split=custom_split,
                                                        min_mag=min_mag,
                                                        mag_key=mag_key,
                                                        overwrite_sampling_rate=overwrite_sampling_rate,
                                                        first_appearance_list=first_appearance_list,
                                                        last_appearance_list=last_appearance_list)

    if args.additional_data:
        print('Loading additional data')
        event_metadata_add, data_add, _ = loader.load_events(args.additional_data,
                                                             parts=(True, True, True),
                                                             min_mag=min_mag,
                                                             mag_key=mag_key,
                                                             overwrite_sampling_rate=overwrite_sampling_rate)
        event_metadata = pd.concat([event_metadata, event_metadata_add])
        for t_key in data.keys():
            if t_key in data_add:
                data[t_key] += data_add[t_key]

    if pga_key in data:
        pga_true = data[pga_key]
    else:
        pga_true = None
        
    if args.n_pga_targets is not None:
        if config['model_params'].get('n_pga_targets', 0) > 0:
            print('Overwriting number of PGA targets')
            config['model_params']['n_pga_targets'] = args.n_pga_targets
        else:
            print('PGA flag is set, but model does not support PGA')

    
    ensemble = config.get('ensemble', 1)
    print(ensemble)
    if ensemble > 1 and not args.ensemble_member:
        experiment_path = args.experiment_path
        model_list = EnsembleEvaluateModel(config, experiment_path, weight_file=experiment_path+'/0/checkpoint_{}.pth'.format(args.weight_file), loss_limit=args.loss_limit, batch_size=batch_size, device=device)

    mag_stats = []
    loc_stats = []
    pga_stats = []
    mag_pred_full = []
    loc_pred_full = []
    pga_pred_full = []
    
    results = {'times': times,
               'pga_stats': np.array(pga_stats).tolist()}

    with open(os.path.join(output_dir, 'stats.json'), 'w') as stats_file:
        json.dump(results, stats_file, indent=4)

    if args.head_times:
        times_pga = np.arange(args.blind_time, 25, 0.8)
        alpha = [float(x) for x in args.alpha.split(',')]
        warning_time_information = calculate_warning_times(config, model_list, data, event_metadata,
                                                           times=times_pga,
                                                           alpha=alpha,
                                                           batch_size=batch_size,
                                                           use_multiprocessing=not args.no_multiprocessing,
                                                           dataset_id=args.dataset_id, device=device)
    else:
        warning_time_information = None
        alpha = None


    with open(os.path.join(output_dir, f'{args.weight_file}_EndSurvival_warning.pkl'), 'wb') as pred_file:
        pickle.dump((times, mag_pred_full, loc_pred_full, pga_pred_full, warning_time_information, alpha), pred_file)
        