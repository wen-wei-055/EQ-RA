import glob
import h5py
import numpy as np
import pandas as pd
import obspy
import os
from obspy import UTCDateTime
import warnings
import time
from tqdm import tqdm
from scipy.stats import norm
import torch
from torch.utils.data import Dataset

D2KM = 111.19492664455874

def resample_trace(trace, sampling_rate):
    if trace.stats.sampling_rate == sampling_rate:
        return
    if trace.stats.sampling_rate % sampling_rate == 0:
        trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
    else:
        trace.resample(sampling_rate)

def detect_location_keys(columns): 
    candidates = [['LAT', 'Latitude(°)', 'Latitude', 'source_latitude_deg'],  
                  ['LON', 'Longitude(°)', 'Longitude', 'source_longitude_deg'],
                  ['DEPTH', 'JMA_Depth(km)', 'Depth(km)', 'Depth/Km', 'source_depth_km']]

    coord_keys = []
    for keyset in candidates:
        for key in keyset:
            if key in columns:
                coord_keys += [key]
                break

    if len(coord_keys) != len(candidates):
        raise ValueError('Unknown location key format')

    return coord_keys

class PreloadedEventGenerator(Dataset):
    def __init__(self, datapath, min_mag, limit, stations_table, stations_channel_boolean, stations_channel_class, latlon_IDtable, topK_nearest_file_path,
                     first_appearance_list, last_appearance_list,
                     key='MA', batch_size=5, cutout=None,sliding_window=False, windowlen=3000, shuffle=True,
                     coords_target=True, oversample=1, pos_offset=(-21, -69),
                     label_smoothing=False, station_blinding=False, magnitude_resampling=3,
                     pga_targets=None, adjust_mean=True, transform_target_only=False,
                     max_stations=None, trigger_based=None, min_upsample_magnitude=2,
                     disable_station_foreshadowing=False, selection_skew=None, pga_from_inactive=False,
                     integrate=False, sampling_rate=100.,
                     select_first=False, fake_borehole=False, scale_metadata=True, pga_key='pga',
                     pga_mode=False, p_pick_limit=5000, coord_keys=None, upsample_high_station_events=None,
                     no_event_token=False, pga_selection_skew=None, **kwargs):
        if kwargs:
            print(f'Unused parameters: {", ".join(kwargs.keys())}')
        self.topK_nearest_file_path = topK_nearest_file_path
        self.data_path = datapath
        self.stations_table = stations_table
        
        self.event_metadata, self.trace_filename, self.survival_table, self.reassigned_pga = self.load_events(datapath, min_mag, limit, first_appearance_list, last_appearance_list)
        
        self.latlon_IDtable = latlon_IDtable
        self.latlon_ID = np.array(list(self.latlon_IDtable.values())) 

        self.stations_channel_class = stations_channel_class
        self.stations_channel_boolean = stations_channel_boolean
        self.batch_size = batch_size 
        self.shuffle = shuffle
        
        self.key = key
        self.cutout = cutout
        self.sliding_window = sliding_window  # If true, selects sliding windows instead of cutout. Uses cutout as values for end of window.
        self.windowlen = windowlen  # Length of window for sliding window
        self.coords_target = coords_target
        self.oversample = oversample  
        self.pos_offset = pos_offset
        self.label_smoothing = label_smoothing
        self.station_blinding = station_blinding
        self.magnitude_resampling = magnitude_resampling
        self.pga_targets = pga_targets
        self.adjust_mean = adjust_mean
        self.transform_target_only = transform_target_only
        if max_stations is None:
            max_stations = batch_waveforms.shape[1]
        self.left_station = max_stations
        self.right_station = len(stations_table)
        self.trigger_based = trigger_based
        self.disable_station_foreshadowing = disable_station_foreshadowing
        self.selection_skew = selection_skew
        self.pga_from_inactive = pga_from_inactive
        self.pga_selection_skew = pga_selection_skew
        self.integrate = integrate
        self.sampling_rate = sampling_rate
        self.select_first = select_first
        self.fake_borehole = fake_borehole
        self.scale_metadata = scale_metadata
        self.upsample_high_station_events = upsample_high_station_events
        self.no_event_token = no_event_token

        # Extend samples to include all pga targets in each epoch
        # PGA mode is only for evaluation, as it adds zero padding to the input/pga target!
        self.pga_mode = pga_mode
        self.p_pick_limit = p_pick_limit

        self.base_indexes = np.arange(len(self.event_metadata))
        self.reverse_index = None
        if magnitude_resampling > 1:
            magnitude = self.event_metadata[key].values
            for i in np.arange(min_upsample_magnitude, 9):
                ind = np.where(np.logical_and(i < magnitude, magnitude <= i + 1))[0]
                self.base_indexes = np.concatenate(
                    (self.base_indexes, np.repeat(ind, int(magnitude_resampling ** (i - 1) - 1))))

        if self.upsample_high_station_events is not None:
            new_indexes = []
            for ind in self.base_indexes:
                n_stations = sum(1 for name in self.trace_filename[ind] if 'HL' in str(name))
                new_indexes += [ind for _ in range(n_stations // self.upsample_high_station_events + 1)]
            self.base_indexes = np.array(new_indexes)
        
        if pga_mode: 
            new_base_indexes = []
            self.reverse_index = []
            c = 0
            for idx in self.base_indexes: 
                # This produces an issue if there are 0 pga targets for an event.
                # As all input stations are always pga targets as well, this should not occur.
                num_samples = 1
                new_base_indexes += [(idx, i) for i in range(num_samples)]
                self.reverse_index += [c]
                c += num_samples
            self.reverse_index += [c]
            self.base_indexes = new_base_indexes

        # self.indexes = np.arange(len(self.event_metadata)) 
        if coord_keys is None:
            self.coord_keys = detect_location_keys(self.event_metadata.columns)
        else:
            self.coord_keys = coord_keys
        
        self.indexes = np.repeat(self.base_indexes.copy(), self.oversample, axis=0)
        with open('indexes.txt','w') as f:
            f.write(str(list(self.indexes)))
        if self.shuffle:
            np.random.shuffle(self.indexes)

        self.total_coords = np.zeros(((len(self.stations_table),3)))
        for coor_i, station_coord in enumerate(list(self.stations_table.keys())):
            self.total_coords[coor_i,:] = station_coord.split(',')[:]

    def __len__(self):
        return int(np.ceil(self.indexes.shape[0] / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        data = {}
        with h5py.File(self.data_path[0], 'r') as f:
            data_file = f['data']
            for event_idx in indexes: 
                event_name = list(self.event_metadata['data_file'])[int(event_idx)]
                g_event = data_file[event_name]
                for key in g_event:
                    if key not in data:
                        data[key] = []
                    data[key] += [g_event[key][()]]

        batch_p_picks = data['p_picks']
        batch_pga = data['pga']
        batch_waveforms = data['waveforms'] 
        batch_metadata = data['coords']
        
        for i in range(len(batch_p_picks)):
            batch_p_picks[i] -= np.min(batch_p_picks[i]) - 500
            
        true_batch_size = len(indexes)
        if self.pga_mode:
            pga_indexes = [x[1] for x in indexes] 
            indexes = [x[0] for x in indexes] 

        waveforms = np.zeros((true_batch_size, self.left_station) + batch_waveforms[0].shape[1:])
        p_picks = np.zeros((true_batch_size, self.left_station))
        survivals = np.zeros((true_batch_size, 488))
        pga = np.zeros((true_batch_size, self.right_station))
        metadata = np.zeros((true_batch_size, self.left_station) + batch_metadata[0].shape[1:])
        for i, idx in enumerate(indexes): 
            for staion_index in range(batch_waveforms[i].shape[0]): # 事件的每個trace迴圈
                station_key = f"{batch_metadata[i][staion_index][0]},{batch_metadata[i][staion_index][1]},{batch_metadata[i][staion_index][2]}"
                position = self.stations_table[station_key]
                waveforms[i,position] = batch_waveforms[i][staion_index]
                p_picks[i,position] = batch_p_picks[i][staion_index]
                
            pga[i] = self.reassigned_pga[idx]
            metadata[i] = self.total_coords
            survivals[i] = self.survival_table[idx] 

        org_waveform_length = waveforms.shape[2]
        if self.cutout:
            if self.sliding_window:
                windowlen = self.windowlen
                window_end = np.random.randint(max(windowlen, self.cutout[0]), min(waveforms.shape[2], self.cutout[1]) + 1)
                waveforms = waveforms[:, :, window_end - windowlen: window_end]

                cutout = window_end
                if self.adjust_mean:
                    waveforms -= np.mean(waveforms, axis=2, keepdims=True)
            else:
                cutout = np.random.randint(*self.cutout)  #cutout = 400, 3000
                if self.adjust_mean:
                    waveforms -= np.mean(waveforms[:, :, :cutout+1], axis=2, keepdims=True) 
                waveforms[:, :, cutout:] = 0 
        else:
            cutout = waveforms.shape[2]

        if self.trigger_based:
            # Remove waveforms for all stations that did not trigger yet to avoid knowledge leakage
            p_picks[p_picks <= 0] = org_waveform_length 
            waveforms[cutout < p_picks, :, :] = 0


        if self.integrate:
            waveforms = np.cumsum(waveforms, axis=2) / self.sampling_rate

        if self.station_blinding:
            mask = np.zeros(waveforms.shape[:2], dtype=bool)

            for i in range(waveforms.shape[0]):
                active = np.where((waveforms[i] != 0).any(axis=(1, 2)))[0]
                if len(active) == 0:
                    active = np.zeros(1, dtype=int)
                blind_length = np.random.randint(0, len(active))
                np.random.shuffle(active)
                blind = active[:blind_length]
                mask[i, blind] = True

            waveforms[mask] = 0
        mask = np.logical_and((metadata == 0).all(axis=(1, 2)), (waveforms == 0).all(axis=(1, 2, 3)))
        waveforms[mask, 0, 0, 0] = 1e-9
        metadata[mask, 0, 0] = 1e-9

        waveforms = torch.from_numpy(waveforms.astype('float32'))
        metadata = torch.from_numpy(metadata.astype('float32'))
        survivals = torch.from_numpy(survivals)
        inputs = [waveforms, metadata, survivals]
        outputs = []

        if self.pga_targets:
            outputs = [torch.from_numpy(pga.astype('float32'))]
        return inputs, outputs
    
    
    def reassigned_label(self, st_pgas, st_coords):  # (stations,) , (stations,3)
        import json
        nearest_topK_stations = json.load(open(self.topK_nearest_file_path,'r'))
        final_st_pgas = np.zeros((249)) - 1
        
        for st_pga, st_coord in zip(st_pgas, st_coords):
            station_key = f"{st_coord[0]},{st_coord[1]},{st_coord[2]}"
            position = self.stations_table[station_key]  # 0~248
            final_st_pgas[position] = st_pga
        
        tmp_st_pgas = final_st_pgas.copy() 
        
        for i, final_st_pga in enumerate(final_st_pgas):
            if final_st_pga == -1:
                topK_nearest = nearest_topK_stations[str(i)]
                topK_nearest_pga = tmp_st_pgas[topK_nearest]
                topK_nearest_pga = topK_nearest_pga[topK_nearest_pga != -1]
                if len(topK_nearest_pga) > 0:
                    final_st_pgas[i] = np.mean(topK_nearest_pga)
                else:
                    final_st_pgas[i] = 0
        return final_st_pgas
    
    
    def load_events(self, data_paths, min_mag, limit, first_appearance_list, last_appearance_list):
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        if len(data_paths) > 1:
            raise NotImplementedError('Loading partitioned data is currently not supported')
        data_path = data_paths[0]

        event_metadata = pd.read_hdf(data_path, 'metadata/event_metadata')
        if min_mag is not None:
            event_metadata = event_metadata[event_metadata['source_magnitude'] >= min_mag]
        for event_key in ['data_file']:
            if event_key in event_metadata.columns:
                break
                
        if limit:
            event_metadata = event_metadata.iloc[:limit]
        print('data_path',data_path)
        trace_filename = []
        survival_table = []
        total_pga_dic = []
        
        with h5py.File(data_path, 'r') as f:
            for event_i, event in tqdm(event_metadata.iterrows(),total=len(event_metadata)):  
                event_name = str(event['data_file'])
                tmp_survival_table = self.station_survival_table(first_appearance_list, last_appearance_list, event_name[:14])
                if survival_table==[]: survival_table = [tmp_survival_table]
                else: survival_table.append(tmp_survival_table)

                g_event = f['data'][event_name]
                trace_filename += [g_event['trace_filename'][()]]
                
                reassigned_pga = self.reassigned_label(g_event['pga'][()], g_event['coords'][()])
                total_pga_dic.append(reassigned_pga)

        return event_metadata, trace_filename, survival_table, total_pga_dic
    

    def station_survival_table(self, first_appearance_list, last_appearance_list, event_time):
        survival_table = np.ones(len(last_appearance_list))

        event_t = UTCDateTime(event_time)
        for idx in range(len(last_appearance_list)):
            first_t = UTCDateTime(first_appearance_list[idx])
            last_t = UTCDateTime(last_appearance_list[idx])
            if event_t < first_t or event_t > last_t:
                survival_table[idx] = 0

        return survival_table


def generator_from_config(config, data, event_metadata, time, batch_size=64, sampling_rate=100, dataset_id=None):
    training_params = config['training_params']

    if dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[dataset_id]
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]

    noise_seconds = generator_params.get('noise_seconds', 5)
    cutout = int(sampling_rate * (noise_seconds + time))
    cutout = (cutout, cutout + 1)

    n_pga_targets = config['model_params'].get('n_pga_targets', 0)
    left_station = config['model_params']['max_stations']
    generator_params['magnitude_resampling'] = 1
    generator_params['batch_size'] = batch_size
    generator_params['transform_target_only'] = generator_params.get('transform_target_only', True)
    generator_params['upsample_high_station_events'] = None
    if generator_params.get('coord_keys', None) is not None:
        raise NotImplementedError('Fixed coordinate keys are not implemented in location evaluation')
    generator_params['translate'] = False

    generator = PreloadedEventGenerator(data=data,
                                        event_metadata=event_metadata,
                                        coords_target=True,
                                        cutout=cutout,
                                        pga_targets=n_pga_targets,
                                        max_stations=left_station,
                                        sampling_rate=sampling_rate,
                                        select_first=True,
                                        shuffle=False,
                                        pga_mode=True,
                                        **generator_params)
    return generator


class CutoutGenerator(Dataset):
    def __init__(self, generator, times, sampling_rate):
        self.generator = generator
        self.times = times
        self.sampling_rate = sampling_rate
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return len(self.generator) * len(self.times)

    def __getitem__(self, index):
        time, batch_id = self.indexes[index]
        cutout = int(self.sampling_rate * (time + 5))
        self.generator.cutout = (cutout, cutout + 1)
        return self.generator[batch_id]

    def on_epoch_end(self):
        self.indexes = []
        for time in self.times:
            self.indexes += [(time, i) for i in range(len(self.generator))]