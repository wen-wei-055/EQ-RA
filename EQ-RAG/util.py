import h5py
import numpy as np
import pandas as pd
from obspy import UTCDateTime
from tqdm import tqdm
import json
import faiss
import torch
from torch.utils.data import Dataset

import heapq
import random 

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
    def __init__(self, datapath, total_datapath, min_mag, limit, stations_table, stations_channel_boolean, stations_channel_class, latlon_IDtable, topK_nearest_file_path, experiment_retrieve,
                     retrieve_event, peek_sample, first_appearance_list, last_appearance_list, district_PGA_event_idx, datatype, RAG_topK, #datafile_2_eventidx,
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
        self.datatype = datatype
        self.RAG_time_PGA_document = np.load('/mnt/disk5/william/Dataset/analysis_file/RAG/vector/time_PGA/total_CosineSimilarity_RAG_time_PGA.npy')
        self.RAGindexes = self.InitialIndex(datatype, self.RAG_time_PGA_document) 
        
        self.experiment_retrieve = experiment_retrieve
        if experiment_retrieve:
            self.retrieve_event = retrieve_event

        # Train: 6027
        # Val: 2998
        # Test: 1367
        if datatype == 'Train':
            self.datatype_ID_diff = 0
        if datatype == 'Val':
            self.datatype_ID_diff = 6027
        self.topK_nearest_file_path = topK_nearest_file_path
        self.topK_num = 2
        self.PGA_level = np.array([0.08, 0.25, 0.8, 1.4, 2.5, 4.4, 8.0])
        self.RAG_topK = RAG_topK
        self.total_datapath = total_datapath
        self.data_path = datapath
        self.district_PGA_event_idx = district_PGA_event_idx
        self.stations_table = stations_table
        self.event_metadata, self.trace_filename, self.survival_table, self.station_event_idx_list, self.reassigned_pga = self.load_events(datapath, min_mag, limit, first_appearance_list, last_appearance_list)
        self.total_event_metadata = pd.read_hdf(total_datapath[0], 'metadata/event_metadata')
        self.latlon_IDtable = latlon_IDtable
        self.latlon_ID = np.array(list(self.latlon_IDtable.values())) 

        self.stations_channel_class = stations_channel_class
        self.stations_channel_boolean = stations_channel_boolean
        self.batch_size = batch_size 
        self.shuffle = shuffle
        
        self.key = key
        self.cutout = cutout
        self.peek_sample = peek_sample
        self.sliding_window = sliding_window  # If true, selects sliding windows instead of cutout. Uses cutout as values for end of window.
        self.windowlen = windowlen  # Length of window for sliding window
        self.coords_target = coords_target
        self.oversample = oversample  # 設1
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
        
        if pga_mode:  #evaluate.py在用的
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
        search_time = cutout // 100  =
        
        # Generate indexes of the batch
        original_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        true_batch_size = len(original_indexes) 
        
        indexes_pairs = [] 
        original_data = {}
        pair_data = {}
        with h5py.File(self.total_datapath[0], 'r') as f:
            data_file = f['data']
            for event_idx in original_indexes:
                station_district_idx = self.station_event_idx_list[event_idx]
                event_idx = event_idx + self.datatype_ID_diff
                event_name = list(self.total_event_metadata['data_file'])[int(event_idx)]

                g_event = data_file[event_name]
                for key in g_event:
                    if key not in original_data:
                        original_data[key] = [] 
                    if key == 'p_picks':
                        tmp_pick = g_event[key][()]
                        tmp_pick -= np.min(tmp_pick) - 500
                        original_data[key]+= [tmp_pick]
                        
                        exclude_indices = np.where(tmp_pick>cutout)
                        tmp_indices = np.argsort(tmp_pick)
                        tmp_indices = np.setdiff1d(tmp_indices, exclude_indices)
                        topK_indices = tmp_indices[:5]
                        topK_districts = station_district_idx[topK_indices]
                        
                        if len(topK_districts) != 0:
                            np.random.shuffle(topK_districts)
                            select_event = self.event_select(topK_districts[0], event_idx, search_time)
                            indexes_pairs.append(select_event) 
                        else:
                            indexes_pairs.append(-1)
                    else:
                        original_data[key] += [g_event[key][()]]

            for indexes_pair in indexes_pairs:
                if indexes_pair == -1:
                    if 'p_picks' not in pair_data.keys():
                        pair_data['p_picks'] = [] 
                        pair_data['pga'] = []
                        pair_data['waveforms'] = []
                        pair_data['coords'] = []
                    pair_data['p_picks'] += [np.zeros((1,))]
                    pair_data['pga'] += [np.zeros((1,))]
                    pair_data['waveforms'] += [np.zeros((1,3000,3))]
                    pair_data['coords'] += [np.zeros((1,3))]
                else:
                    event_name = list(self.total_event_metadata['data_file'])[int(indexes_pair)]
                    g_event = data_file[event_name]
                    for key in g_event:
                        if key not in pair_data:
                            pair_data[key] = [] 
                        if key == 'p_picks':
                            tmp_pick = g_event[key][()]
                            tmp_pick -= np.min(tmp_pick) - 500
                            pair_data[key]+= [tmp_pick]
                        else:
                            pair_data[key] += [g_event[key][()]]

        datas = [original_data, pair_data]
        pairs = [original_indexes, indexes_pairs]
            
        total_waveforms = np.zeros((true_batch_size, self.left_station*self.topK_num, 3000, 3))
        total_pga = np.zeros((true_batch_size, self.right_station))
        left_survival = np.zeros((true_batch_size, self.right_station))
        total_metadata = np.tile(self.total_coords, (true_batch_size,2,1))
        
        for ii, (indexes, data) in enumerate(zip(pairs,datas)):
            batch_p_picks = data['p_picks']
            batch_waveforms = data['waveforms']  
            batch_metadata = data['coords']
            
            if self.pga_mode:
                pga_indexes = [x[1] for x in indexes] 
                indexes = [x[0] for x in indexes]  

            waveforms = np.zeros((true_batch_size, self.left_station) + batch_waveforms[0].shape[1:])
            p_picks = np.zeros((true_batch_size, self.left_station))

            survivals = np.zeros((true_batch_size, 488))
            pga = np.zeros((true_batch_size, self.right_station))
            
            for i, idx in enumerate(indexes): 
                if idx == -1: 
                    continue
                for staion_index in range(batch_waveforms[i].shape[0]): 
                    station_key = f"{batch_metadata[i][staion_index][0]},{batch_metadata[i][staion_index][1]},{batch_metadata[i][staion_index][2]}"
                    position = self.stations_table[station_key]
                    waveforms[i,position] = batch_waveforms[i][staion_index]
                    p_picks[i,position] = batch_p_picks[i][staion_index]
                if ii == 0:
                    pga[i] = self.reassigned_pga[idx]
                    survivals[i] = self.survival_table[idx] 
                
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
                total_pga[:,:self.left_station] = pga
                left_survival = survivals
            else:
                total_waveforms[:,self.left_station:,:,:] = waveforms
        
        total_waveforms = torch.from_numpy(total_waveforms.astype('float32'))
        total_metadata = torch.from_numpy(total_metadata.astype('float32'))
        left_survival = torch.from_numpy(left_survival)
        inputs = [total_waveforms, total_metadata, left_survival]
        outputs = []
        if self.pga_targets:
            outputs = [torch.from_numpy(total_pga.astype('float32'))]
        return inputs, outputs
    
    def InitialIndex(self, dtype, doc):
        # Train: 6027
        # Val: 2998
        # Test: 1367
        # index:
        # 0~6026 / 6027~9024 / 9025~10391
        
        indexes = []
        for n in range(31):
            index = faiss.IndexFlatIP(doc.shape[-2])
            index.add(doc[:6027,:,n])
            indexes.append(index)
        return indexes
    
    def event_select(self, district, EventID, query_time):
        if self.experiment_retrieve:
            if self.experiment_retrieve == 'MAE':
                EventID = EventID - self.datatype_ID_diff
                label_event_relationship = self.retrieve_event[str(EventID)]
                label_event_relationship = {k: v for k, v in label_event_relationship.items() if v != 0} 
                
                # Get the three smallest items (keys) from label_event_relationship
                three_smallest_items = heapq.nsmallest(self.RAG_topK, label_event_relationship.items(), key=lambda item: item[1])

                min_keys = [int(key) for key, value in three_smallest_items]
                random_key = random.choice(min_keys)
                return random_key 
                
        else:
            query = np.expand_dims(self.RAG_time_PGA_document[EventID,:,query_time], 0)
            distance_list, I = self.RAGindexes[query_time].search(query, self.RAG_topK)  
            I = I[0]

            if self.RAG_topK > 1:
                np.random.shuffle(I)
                
            I = I[0]
            return I  
    
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
        station_2_district = json.load(open('RAG_configs/idx_coords_fault_249_table.json','r'))
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        if len(data_paths) > 1:
            raise NotImplementedError('Loading partitioned data is currently not supported')
        data_path = data_paths[0]

        event_metadata = pd.read_hdf(data_path, 'metadata/event_metadata')
        print('event_metadata',event_metadata['source_origin_time'])
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
        station_event_idx_list = []
        total_pga_dic = []
        
        with h5py.File(data_path, 'r') as f:
            for _, event in tqdm(event_metadata.iterrows(),total=len(event_metadata)):  
                event_name = str(event['data_file'])
                tmp_survival_table = self.station_survival_table(first_appearance_list, last_appearance_list, event_name[:14])
                if survival_table==[]: survival_table = [tmp_survival_table]
                else: survival_table.append(tmp_survival_table)
                
                g_event = f['data'][event_name]
                for key in g_event:
                    if key == 'trace_filename':
                        trace_filename += [g_event[key][()]]
                    if key == 'coords':
                        tmp_station_idx = []
                        for coord in g_event[key][()]:
                            tmp_station_idx.append(station_2_district[f'{coord[0]},{coord[1]},{coord[2]}'])
                        tmp_station_idx = np.array(tmp_station_idx)
                        station_event_idx_list.append(tmp_station_idx)
                        
                reassigned_pga = self.reassigned_label(g_event['pga'][()], g_event['coords'][()])
                total_pga_dic.append(reassigned_pga)
                        
        return event_metadata, trace_filename, survival_table, station_event_idx_list, total_pga_dic

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
