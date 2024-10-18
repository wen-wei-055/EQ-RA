import numpy as np
import h5py
import torch
from tqdm import tqdm
import os
import pickle
import argparse
import json
import time
from scipy.stats import norm
import sys

import util
import loader
import models
import logging
import shutil

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_np_pt(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed) #CPU seed
    torch.cuda.manual_seed(seed) #GPU seed


def transfer_weights(model, weights_path, ensemble_load=False, wait_for_load=False, ens_id=None, sleeptime=600):
    print("weights_path")
    if ensemble_load:
        weights_path = os.path.join(weights_path, f'{ens_id}')

    # If weight file does not exists, wait until it exists. Intended for ensembles. Warning: Can deadlock program.
    if wait_for_load:
        if os.path.isfile(weights_path):
            target_object = weights_path
        else:
            target_object = os.path.join(weights_path, 'train.log')

        while not os.path.exists(target_object):
            print(f'File {target_object} for weight transfer missing. Sleeping for {sleeptime} seconds.')
            time.sleep(sleeptime)

    if os.path.isdir(weights_path):
        last_weight = sorted([x for x in os.listdir(weights_path) if x[:11] == 'checkpoint_'])[-1] 
        weights_path = os.path.join(weights_path, last_weight)
        
    print(weights_path)
    own_state = model.state_dict()
    state_dict = torch.load(weights_path)['model_weights']
    
    for name, param in state_dict.items():
        if name not in own_state.keys():
            print(f"{name} is not load weight")
            continue
        else:
            print(name)
            own_state[name].copy_(param)
            
    full_model.load_state_dict(own_state)
    return full_model


def gaussian_confusion_matrix(type, status, confusion_matrix, targets_pga=None, pred=None, thresholds=None, loop=None, total_loss=None, optimizer=None):
    if status == 'accumulate':
        pred_matrix = np.empty((pred.shape[0], len(thresholds)))
        targets_pga = np.reshape(targets_pga,(targets_pga.shape[0],1))
        targets_pga = (targets_pga >= thresholds).astype(int)
        targets_pga = np.sum(targets_pga, axis=1)
        for j, level in enumerate(thresholds):
            prob = np.sum(
                pred[:, :, 0] * (1 - norm.cdf((level - pred[:, :, 1]) / pred[:, :, 2])),
                axis=-1) 
            exceedance = prob >= 0.2
            pred_matrix[:,j] = exceedance
        pred_matrix = pred_matrix.astype(int)
        pred_matrix = np.sum(pred_matrix, axis=1)
        for idx in range(len(pred_matrix)):
            confusion_matrix[pred_matrix[idx]][targets_pga[idx]] += 1
            
    elif status == 'write_txt':
        print(confusion_matrix)
        with open(os.path.join(training_params['weight_path'], '{}_confusion_matrix.txt'.format(type)), 'a+', encoding='utf-8') as f:
            f.write(str(epoch)+'\n')
            f.write(str(confusion_matrix)+'\n\n')
            f.write('loss: {},   lr: {}'.format((total_loss / len(loop)), optimizer.param_groups[0]['lr'])+'\n\n')


def training(model, optimizer, loader, epoch, epochs, device, training_params, pga_loss, train_loss_record, logger):
    train_loop = tqdm(loader)
    model.train()
    total_train_loss = 0.0

    thresholds = np.log10(np.array([0.25, 0.8, 1.4, 2.5, 4.4, 8.0])*10)
    confusion_matrix = np.zeros((len(thresholds)+1, len(thresholds)+1)).astype(np.int32)
    for x,y in train_loop:
        inputs_waveforms, inputs_coords, survival_list, targets_pga = \
                x[0].to(device), x[1].to(device), x[2].to(device).long(), y[0].to(device)
        
        survival_list = survival_list[:,-249:]

        targets_0 = targets_pga==0
        targets_pga[targets_0] = 1e-6
        targets_pga[~targets_0] = torch.log10(targets_pga[~targets_0]*10)

        targets_pga = targets_pga.contiguous().view(-1)

        pred = model(inputs_waveforms, inputs_coords, survival_list, stations_channel_class)   # (batch, 儀器數, 10, 3)
        pred = pred.contiguous().view(-1, pred.shape[-2], pred.shape[-1])
        
        survival_list = survival_list.contiguous().view(-1)
        selected_indices = torch.nonzero(survival_list==1, as_tuple=False).squeeze(dim=1)
        
        pred = pred[selected_indices]
        targets_pga = targets_pga[selected_indices]
        
        targets_pga = torch.unsqueeze(targets_pga, 0) 
        pred = torch.unsqueeze(pred,0)
        train_loss = pga_loss(targets_pga, pred)*2    # Find the Loss
        total_train_loss = train_loss.item() + total_train_loss
        train_loss.backward()     # Calculate gradients 
        
        clip_grad_norm_(model.parameters(), training_params['clipnorm'])
        optimizer.step()          # Update Weights     
        optimizer.zero_grad()     # Clear the gradients

        train_loop.set_description(f"[Train Epoch {epoch+1}/{epochs}]")
        train_loop.set_postfix(loss=train_loss.detach().cpu().item())
        
        ######################### confusion matrix #########################
        targets_pga = torch.squeeze(targets_pga, 0)
        pred = torch.squeeze(pred, 0)

        targets_pga = targets_pga.contiguous().view(-1, (pred.shape[-1] - 1) // 2).cpu().numpy()   #(batch, 20, 1, 1) -> (batch*20, 1, 1)
        pred = pred.contiguous().view(-1, pred.shape[-2], pred.shape[-1]).detach().cpu().numpy() #(batch, 20, 5, 3) -> (batch*20, 5, 3)
        gaussian_confusion_matrix('train', 'accumulate', confusion_matrix, targets_pga=targets_pga, pred=pred, thresholds=thresholds)
    gaussian_confusion_matrix('train', 'write_txt', confusion_matrix, loop=train_loop, total_loss=total_train_loss, optimizer=optimizer)
    

    train_loss_record.append(total_train_loss / len(train_loop))  
    
    logger.info('[Train] epoch: %d -> loss: %.4f' %(epoch, total_train_loss / len(train_loop)))
    return model, optimizer,train_loss_record


def validating(model, optimizer, loader, epoch, epochs, device, pga_loss, scheduler, val_loss_record, logger):
    valid_loop = tqdm(loader)
    model.eval()
    total_val_loss = 0.0
    
    thresholds = np.log10(np.array([0.25, 0.8, 1.4, 2.5, 4.4, 8.0])*10)
    confusion_matrix = np.zeros((len(thresholds)+1, len(thresholds)+1)).astype(np.int32)
    with torch.no_grad():
        for x,y in valid_loop:
            inputs_waveforms, inputs_coords, survival_list, targets_pga = \
                    x[0].to(device), x[1].to(device), x[2].to(device).long(), y[0].to(device)
                    
            survival_list = survival_list[:,-249:]
            
            targets_0 = targets_pga==0
            targets_pga[targets_0] = 1e-6
            targets_pga[~targets_0] = torch.log10(targets_pga[~targets_0]*10)
            
            targets_pga = targets_pga.contiguous().view(-1)
            
            pred = model(inputs_waveforms, inputs_coords, survival_list, stations_channel_class)     
            pred = pred.contiguous().view(-1, pred.shape[2], pred.shape[3] ) # 0,1維度合併
                    
            survival_list = survival_list.contiguous().view(-1)
            selected_indices = torch.nonzero(survival_list==1, as_tuple=False).squeeze(dim=1)
            
            pred = pred[selected_indices]
            targets_pga = targets_pga[selected_indices]
            
            targets_pga = torch.unsqueeze(targets_pga,0)
            pred = torch.unsqueeze(pred,0) 
            val_loss = pga_loss(targets_pga, pred) 
            
            total_val_loss = val_loss.item() + total_val_loss
            valid_loop.set_description(f"[Eval Epoch {epoch+1}/{epochs}]")
            valid_loop.set_postfix(loss=val_loss.detach().cpu().item())

            ######################### confusion matrix #########################
            targets_pga = torch.squeeze(targets_pga, 0)
            pred = torch.squeeze(pred, 0)
            
            targets_pga = targets_pga.contiguous().view(-1, (pred.shape[-1] - 1) // 2).cpu().numpy()   #(batch, 20, 1, 1) -> (batch*20, 1, 1)
            pred = pred.contiguous().view(-1, pred.shape[-2], pred.shape[-1]).detach().cpu().numpy() #(batch, 20, 5, 3) -> (batch*20, 5, 3)

            gaussian_confusion_matrix('val', 'accumulate', confusion_matrix, targets_pga=targets_pga, pred=pred, thresholds=thresholds)
        gaussian_confusion_matrix('val', 'write_txt', confusion_matrix, loop=valid_loop, total_loss=total_val_loss, optimizer=optimizer)
        

    val_loss_record.append((total_val_loss / len(valid_loop)))
    scheduler.step(total_val_loss/len(valid_loop))
    
    logger.info('[Eval] epoch: %d -> loss: %.4f' %(epoch, total_val_loss/ len(valid_loop)))
    logger.info('======================================================')
    return val_loss_record,scheduler


def my_custom_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
                    "%(lineno)d — %(message)s")
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def latlon_ID(stations_table):
    appear_first = {} 
    appear_dic = {}  
    i = 0
    for id_name, instrument_name in enumerate(stations_table):
        latlon = ','.join(instrument_name.split(',')[:2])

        if latlon not in appear_first.keys():
            appear_first[latlon] = i
            appear_dic[id_name] = i
            i += 1
        if latlon in appear_first.keys():
            appear_dic[id_name] = appear_first[latlon]
    return appear_dic


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
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--test_run', action='store_true')  # Test run with less data
    parser.add_argument('--continue_ensemble', action='store_true')  # Continues a stopped ensemble training
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))

    torch.set_num_threads(6)
    seed_np_pt(config.get('seed', 42))

    training_params = config['training_params']
    generator_params = training_params.get('generator_params', [training_params.copy()])
    device = torch.device(training_params['device'] if torch.cuda.is_available() else "cpu")
    stations_table = json.load(open(training_params['station_json_file'], 'r'))

    latlon_IDtable = latlon_ID(stations_table)
    latlon_IDtable = latlondep_ID(stations_table)

    stations_channel = ['HL' for station_i in list(stations_table.keys())]
    stations_channel_boolean = [1 if item == 'HL' else 0 for item in stations_channel]
    stations_channel_class = [1 if item == 'HL' else 2 if item == 'HH' else 0 for item in stations_channel]
    
    if not os.path.isdir(training_params['weight_path']): os.mkdir(training_params['weight_path'])
    listdir = os.listdir(training_params['weight_path'])

    with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    for doc_name in ['train.py','util.py','models.py','loader.py']:
        shutil.copyfile(doc_name, training_params['weight_path']+'/'+doc_name)

    print('Loading data')
    if args.test_run: limit = 100
    else: limit = None

    assert len(generator_params) == len(training_params['train_data_path'])
    assert len(generator_params) == len(training_params['val_data_path'])

    with open(training_params['station_first_appearance'],'r') as f: first_appearance_list = json.load(f)
    with open(training_params['station_last_appearance'],'r') as f: last_appearance_list = json.load(f)
    first_appearance_list = list(first_appearance_list.values())
    last_appearance_list = list(last_appearance_list.values())

    max_stations = config['model_params']['max_stations']
    ensemble = config.get('ensemble', 1)

    super_config = config.copy()
    super_training_params = training_params.copy()
    super_model_params = config['model_params'].copy()

    for ens_id in [0]:
        print('==============================================================')
        print('===============     第 {}/{} 輪Ensemble開始     ==============='.format(ens_id + 1, ensemble))
        print('==============================================================')
        print(' ')
        if ensemble > 1:
            seed_np_pt(ens_id)

            config = super_config.copy()
            config['ens_id'] = ens_id
            training_params = super_training_params.copy()
            training_params['weight_path'] = os.path.join(training_params['weight_path'], f'{ens_id}')
            config['training_params'] = training_params
            config['model_params'] = super_model_params.copy()

            if training_params.get('ensemble_rotation', False):
                # Rotated by angles between 0 and pi/4
                config['model_params']['rotation'] = np.pi / 4 * ens_id / (ensemble - 1)
            
            if args.continue_ensemble and os.path.isdir(training_params['weight_path']):
                hist_path = os.path.join(training_params['weight_path'], 'hist.pkl')
                if os.path.isfile(hist_path):
                    continue
                else:
                    raise ValueError(f'Can not continue unclean ensemble. Checking for {hist_path} failed.')

            if not os.path.isdir(training_params['weight_path']):
                os.mkdir(training_params['weight_path'])

            with open(os.path.join(training_params['weight_path'], 'config.json'), 'w') as f:
                json.dump(config, f, indent=4)

        full_model = models.build_transformer_model(**config['model_params'], device=device, trace_length=3000).to(device)

        sampling_rate = 100

        noise_seconds = generator_params[0].get('noise_seconds', 5)
        cutout = (
            sampling_rate * (noise_seconds + generator_params[0]['cutout_start']), sampling_rate * (noise_seconds + generator_params[0]['cutout_end']))
        sliding_window = generator_params[0].get('sliding_window', False)
        n_pga_targets = config['model_params'].get('n_pga_targets', 0)
        
        if 'load_model_path' in training_params:
            print('Loading full model')
            full_model.load_weights(training_params['load_model_path'])

        if 'transfer_model_path' in training_params:
            print('Transfering model weights')
            ensemble_load = training_params.get('ensemble_load', False)
            wait_for_load = training_params.get('wait_for_load', False)
            full_model = transfer_weights(full_model, training_params['transfer_model_path'],
                            ensemble_load=ensemble_load, wait_for_load=wait_for_load, ens_id=ens_id)
        
        train_datas = []
        val_datas = []               
        
        for i, generator_param_set in enumerate(generator_params): 
            noise_seconds = generator_param_set.get('noise_seconds', 5)
            cutout = (sampling_rate * (noise_seconds + generator_param_set['cutout_start']), sampling_rate * (noise_seconds + generator_param_set['cutout_end']))

            generator_param_set['transform_target_only'] = generator_param_set.get('transform_target_only', True)
            train_datas += [util.PreloadedEventGenerator(datapath=training_params['train_data_path'],
                                                        topK_nearest_file_path=training_params['topK_nearest_file'],
                                                        min_mag=generator_param_set.get('min_mag', 0),
                                                        limit=limit,
                                                        stations_table=stations_table,
                                                        coords_target=True,
                                                        label_smoothing=True,
                                                        station_blinding=True,
                                                        cutout=cutout,
                                                        pga_targets=n_pga_targets,
                                                        max_stations=max_stations,
                                                        sampling_rate=sampling_rate,
                                                        stations_channel_boolean=stations_channel_boolean,
                                                        stations_channel_class=stations_channel_class,
                                                        latlon_IDtable=latlon_IDtable,
                                                        first_appearance_list=first_appearance_list,
                                                        last_appearance_list=last_appearance_list,
                                                        **generator_param_set)]
            old_oversample = generator_param_set.get('oversample', 1)
            val_datas += [util.PreloadedEventGenerator(datapath=training_params['val_data_path'],
                                                            topK_nearest_file_path=training_params['topK_nearest_file'],
                                                            min_mag=generator_param_set.get('min_mag', None),
                                                            limit=limit,
                                                            stations_table=stations_table,
                                                            coords_target=True,
                                                            station_blinding=True,
                                                            cutout=cutout,
                                                            pga_targets=n_pga_targets,
                                                            max_stations=max_stations,
                                                            sampling_rate=sampling_rate,
                                                            stations_channel_boolean=stations_channel_boolean,
                                                            stations_channel_class=stations_channel_class,
                                                            latlon_IDtable=latlon_IDtable,
                                                            first_appearance_list=first_appearance_list,
                                                            last_appearance_list=last_appearance_list,
                                                            **generator_param_set)]
            generator_param_set['oversample'] = old_oversample
            
        filepath = os.path.join(training_params['weight_path'], 'event-{epoch:02d}.hdf5')
        workers = training_params.get('workers', 0)
        
        
        optimizer = torch.optim.Adam(full_model.parameters(), lr=training_params['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5)
        
        def pga_loss(y_true, y_pred):
            return models.time_distributed_loss(y_true, y_pred, models.mixture_density_loss, weight=training_params['weighted_loss'], device=device, mean=True, kwloss={'mean': False})
        
        losses = {}
        losses['pga'] = pga_loss
            
        
        num_epochs = training_params['epochs_full_model']
        metrics_record = {}
        train_loss_record = []
        val_loss_record = []
        lr_record = []
        log_path = training_params['weight_path']+'/train.log'
        logger = my_custom_logger(log_path)
        logger.info('start training')
        
        train_generators = DataLoader(train_datas[0], shuffle=True, batch_size=None, collate_fn=models.my_collate, pin_memory=False, num_workers=workers)
        val_generators = DataLoader(val_datas[0], shuffle=False, batch_size=None, collate_fn=models.my_collate, pin_memory=False, num_workers=workers)
            
        for epoch in range(num_epochs):

            full_model, optimizer,train_loss_record = training(full_model, optimizer, train_generators, epoch,num_epochs, device,training_params ,pga_loss,train_loss_record,logger)
            val_loss_record,scheduler = validating(full_model, optimizer, val_generators, epoch,num_epochs, device,pga_loss,scheduler,val_loss_record,logger)
            lr_record.append(scheduler.optimizer.param_groups[0]['lr'])
            
            #save model
            if epoch < 50:
                after_50_min = -999  
            else:
                after_50_min = min(val_loss_record[49:-1])
                    
            if epoch%9==0 or (epoch>49 and epoch<61) or val_loss_record[-1]<after_50_min: 
                metrics_record['train_loss'] = train_loss_record
                metrics_record['val_loss'] = val_loss_record
                metrics_record['lr_record'] = lr_record
                with open (os.path.join(training_params['weight_path'], 'metrics.txt'), 'w', encoding='utf-8') as f:
                    f.write(str(metrics_record))

                print("-----Saving checkpoint-----")
                torch.save({
                    'model_weights' : full_model.state_dict(), 
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                            }, 
                os.path.join(training_params['weight_path'], f'checkpoint_{epoch:02d}.pth'))
        
            
        
              