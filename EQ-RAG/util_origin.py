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
import faiss
import torch
from torch.utils.data import Dataset

D2KM = 111.19492664455874
import numpy as np
import json
from scipy.stats import norm
import faiss
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

#keras.utils.Sequence 批次載入input data

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
    def __init__(self, event_id, data, data_total, event_metadata, stations_table, stations_channel_boolean, stations_channel_class, topK_nearest_file_path,
                     retrieve_event, experiment_retrieve, district_PGA_event_idx, validation_set, peek_sample, all_station=False,key='MA', batch_size=32, cutout=None,
                     sliding_window=False, windowlen=3000, shuffle=True,
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
        
        self.experiment_retrieve = experiment_retrieve
        if experiment_retrieve:
            self.retrieve_event = retrieve_event
            
        self.topK_nearest_file_path = topK_nearest_file_path
        self.topK_num = 2
        self.validation_set = validation_set
        self.RAG_time_PGA_document = np.load('/mnt/disk5/william/Dataset/analysis_file/RAG/vector/time_PGA/total_CosineSimilarity_RAG_time_PGA.npy')
        self.RAGindexes = self.InitialIndex(self.RAG_time_PGA_document)  

        self.total_waveforms = data_total['waveforms'] 
        self.total_p_picks = data_total['p_picks']  
        self.total_station_coords = data_total['coords']
        self.total_survival_table = data_total['survival_table']
        self.event_id = event_id
        self.peek_sample = peek_sample

        self.district_PGA_event_idx = district_PGA_event_idx
        self.pga_times = data['pga_times']
        self.pga = data['pga']
        self.pgv = data['pgv'] 
        self.waveforms = data['waveforms'] 
        self.metadata = data['coords']
        self.trace_filename = data['trace_filename']
        self.survival_table = data['survival_table']
        self.station_event_idx = data['station_event_idx']
        
        if 'p_picks' in data:
            self.p_picks = data['p_picks'] 
        else:
            print('Found no picks')
            self.p_picks = [np.zeros(x.shape[0]) for x in self.waveforms]

        for i in range(len(self.p_picks)):
            self.p_picks[i] -= np.min(self.p_picks[i]) - 500
            
        self.stations_channel_class = stations_channel_class
        self.stations_channel_boolean = stations_channel_boolean
        self.stations_table = stations_table
        self.batch_size = 1 
        self.shuffle = shuffle
        self.event_metadata = event_metadata
        self.PGA_level = np.array([0.08, 0.25, 0.8, 1.4, 2.5, 4.4, 8.0])
        
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
            max_stations = self.waveforms.shape[1]
        self.left_station = 249
        self.right_station = 249
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

        self.reassigned_pga = self.reassigned_label(self.pga, self.metadata)

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

        if coord_keys is None: 
            self.coord_keys = detect_location_keys(self.event_metadata.columns)
        else:
            self.coord_keys = coord_keys
        
        self.indexes = np.repeat(self.base_indexes.copy(), self.oversample, axis=0)
        if self.shuffle:
            np.random.shuffle(self.indexes)

        self.total_coords = np.zeros(((len(self.stations_table),3)))
        for coor_i, station_coord in enumerate(list(self.stations_table.keys())):
            self.total_coords[coor_i,:] = station_coord.split(',')[:]

    def __len__(self):
        return int(np.ceil(self.indexes.shape[0] / self.batch_size))

    def __getitem__(self, index): 
        cutout = np.random.randint(*self.cutout)
        station_district_idx = self.station_event_idx[0]
        search_time = cutout // 100 
        
        tmp_p_picks = self.p_picks[0] 
        
        exclude_indices = np.where(tmp_p_picks>cutout)
        tmp_indices = np.argsort(tmp_p_picks)
        tmp_indices = np.setdiff1d(tmp_indices, exclude_indices) 
        top1_indices = tmp_indices[0]
        top1_districts = station_district_idx[top1_indices]
        select_event, topK_list = self.event_select(top1_districts, self.event_id, search_time)
        if select_event == -1:
            pair_waveforms = np.zeros(self.total_waveforms[self.event_id].shape)
            pair_metadata = np.zeros(self.total_station_coords[self.event_id].shape)
            pair_picks = np.zeros(self.total_p_picks[self.event_id].shape)
        else:
            pair_waveforms = self.total_waveforms[select_event]
            pair_metadata = self.total_station_coords[select_event]
            pair_picks = self.total_p_picks[select_event]
        
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]        
        true_batch_size = len(indexes) 
        if self.pga_mode:
            pga_indexes = [x[1] for x in indexes] 
            indexes = [x[0] for x in indexes] 

        datas = [self.waveforms, self.metadata, self.p_picks, self.survival_table]
        pairs = [[pair_waveforms], [pair_metadata], [pair_picks], None]

        iter_list = [datas, pairs]
        iter_index_list = [indexes, [select_event]]

        total_waveforms = np.zeros((true_batch_size, self.left_station*self.topK_num, 3000, 3))
        survivals = np.zeros((true_batch_size, 488))
        total_metadata = np.tile(self.total_coords, (true_batch_size,2,1))
        pga = np.zeros((true_batch_size, 249))

        
        for ii, (indexes,data) in enumerate(zip(iter_index_list, iter_list)):
            
            tmp_waveforms = data[0]
            tmp_metadata = data[1]
            tmp_p_picks = data[2]
            tmp_survival_table = data[3]
            
            waveforms = np.zeros((true_batch_size, self.left_station) + self.waveforms[0].shape[1:])
            p_picks = np.zeros((true_batch_size, self.left_station))

            for i, idx in enumerate(indexes): 
                if idx == -1: 
                    continue
                
                for staion_index in range(tmp_waveforms[i].shape[0]): 
                    station_key = f"{tmp_metadata[0][staion_index][0]},{tmp_metadata[0][staion_index][1]},{tmp_metadata[0][staion_index][2]}"
                    position = self.stations_table[station_key]
                    waveforms[i,position] = tmp_waveforms[i][staion_index]
                    p_picks[i,position] = tmp_p_picks[i][staion_index]
                
                if ii == 0:
                    pga[i] = self.reassigned_pga[idx]
                    survivals[i] = tmp_survival_table[idx]  

            org_waveform_length = waveforms.shape[2]
            if ii == 0:
                waveforms[:, :, cutout:] = 0
            else:
                waveforms[:, :, cutout+self.peek_sample:] = 0 

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
            
            if ii==0:
                total_waveforms[:,:self.left_station,:,:] = waveforms
            else:
                total_waveforms[:,self.left_station:,:,:] = waveforms
                
        total_waveforms = torch.from_numpy(total_waveforms.astype('float32'))
        total_metadata = torch.from_numpy(total_metadata.astype('float32'))
        pga = torch.from_numpy(pga.astype('float32'))
        survivals = torch.from_numpy(survivals)
        inputs = [total_waveforms, total_metadata, survivals]
        outputs = [pga, topK_list]

        return inputs, outputs
    
    
    def reassigned_label(self, st_pgas, st_coords):  # (stations,) , (stations,3)
        import json
        nearest_topK_stations = json.load(open(self.topK_nearest_file_path,'r'))
        assert len(st_pgas) == len(st_coords)
        final_st_pgas_list = []
        
        for event_i in range(len(st_pgas)):
            final_st_pgas = np.zeros((249)) - 1
            for st_pga, st_coord in zip(st_pgas[event_i], st_coords[event_i]):
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
            
            final_st_pgas_list.append(final_st_pgas)
        return final_st_pgas_list
    
    def InitialIndex(self, doc):
        indexes = []
        for n in range(31):
            index = faiss.IndexFlatIP(doc.shape[-2])
            index.add(doc[:6027,:,n])
            indexes.append(index)
        return indexes
    
    def event_select(self, district, EventID, query_time):
        if self.experiment_retrieve:
            if self.experiment_retrieve == 'MAE':
                EventID = EventID - 9025
                label_event_relationship = self.retrieve_event[str(EventID)]
                label_event_relationship = {k: v for k, v in label_event_relationship.items() if v != 0} 
                if label_event_relationship.items():
                    min_key, min_value = min(label_event_relationship.items(), key=lambda item: item[1])
                    min_key = int(min_key)
                else:
                    min_key = -1
                return min_key, np.zeros((10,)) 
                
        else:
            query = np.expand_dims(self.RAG_time_PGA_document[EventID,:,query_time], 0)
            _, I = self.RAGindexes[query_time].search(query, 10)  
            I_topK = np.array(I[0])
            I = I[0][0]
            return I, I_topK   


def generator_from_config(config, args, data, event_metadata, time, batch_size=64, sampling_rate=100, dataset_id=None):
    training_params = config['training_params']

    if dataset_id is not None:
        generator_params = training_params.get('generator_params', [training_params.copy()])[dataset_id]
    else:
        generator_params = training_params.get('generator_params', [training_params.copy()])[0]

    noise_seconds = generator_params.get('noise_seconds', 5)
    cutout = int(sampling_rate * (noise_seconds + time))
    cutout = (cutout, cutout + 1)

    n_pga_targets = config['model_params'].get('n_pga_targets', 0)
    max_stations = config['model_params']['max_stations']
    generator_params['magnitude_resampling'] = 1
    generator_params['batch_size'] = batch_size
    generator_params['transform_target_only'] = generator_params.get('transform_target_only', True)
    generator_params['upsample_high_station_events'] = None
    if generator_params.get('coord_keys', None) is not None:
        raise NotImplementedError('Fixed coordinate keys are not implemented in location evaluation')
    generator_params['translate'] = False

    generator = PreloadedEventGenerator(data=data,
                                        validation_set=args.val,
                                        event_metadata=event_metadata,
                                        coords_target=True,
                                        cutout=cutout,
                                        pga_targets=n_pga_targets,
                                        max_stations=max_stations,
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

