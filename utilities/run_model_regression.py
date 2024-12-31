import torch
import time

from .constants import *
from utilities.device import get_device
from .lr_scheduling import get_lr
import torch.nn.functional as F
import random

def train_epoch(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, print_modulus=1):
    out = -1
    model.train()
    for batch_num, batch in enumerate(dataloader):
        time_before = time.time()
        opt.zero_grad()

        feature_semantic_list = batch["semanticList"].to(get_device())

        feature_scene_offset = batch["scene_offset"].to(get_device())
        feature_motion = batch["motion"].to(get_device())
        feature_emotion = batch["emotion"].to(get_device())

        feature_note_density = batch["note_density"].to(get_device())
        feature_loudness = batch["loudness"].to(get_device())
        feature_instrument = batch["instrument"].to(get_device()).float()

        # Loudness_notedensity and Instrument
        ln_nd, inst = model(feature_semantic_list, 
                            feature_scene_offset,
                            feature_motion,
                            feature_emotion)
        
        ln_nd   = ln_nd.reshape(ln_nd.shape[0] * ln_nd.shape[1], -1)
        
        feature_loudness = feature_loudness.flatten().reshape(-1,1) # (batch_size, 300, 1)
        feature_note_density = feature_note_density.flatten().reshape(-1,1) # (batch_size, 300, 1)
        feature_combined = torch.cat((feature_note_density, feature_loudness), dim=1) # (batch_size, 300, 2)

        # total_loss = loss.forward(ln_nd, feature_combined) + F.binary_cross_entropy(inst, feature_instrument)
        
        # DROPLOSS
        # ln, nd = torch.split(ln_nd, split_size_or_sections=1, dim=1)
        p = random.random()
        if p < 0.8:
            total_loss = loss.forward(ln_nd, feature_combined) + F.binary_cross_entropy(inst, feature_instrument)
        elif p < 0.9:
            total_loss = loss.forward(ln_nd, feature_combined)
        else:
            total_loss = F.binary_cross_entropy(inst, feature_instrument)

        total_loss.backward()
        opt.step()
        
        if(lr_scheduler is not None):
            lr_scheduler.step()
        time_after = time.time()
        time_took = time_after - time_before
        
        if((batch_num+1) % print_modulus == 0):
            print(SEPERATOR)
            print("Epoch", cur_epoch, " Batch", batch_num+1, "/", len(dataloader))
            print("LR:", get_lr(opt))
            print("Train loss:", float(out))
            print("")
            print("Time (s):", time_took)
            print(SEPERATOR)
            print("")
    return

def eval_model(model, dataloader):
    model.eval()
    
    avg_rmse     = -1
    avg_rmse_note_density     = -1
    avg_rmse_loudness     = -1
    with torch.set_grad_enabled(False):
        n_test      = len(dataloader)
        
        sum_total_loss    = 0.0
        sum_rmse_note_density = 0.0
        sum_rmse_loudness = 0.0
        sum_bce_instrument = 0.0

        for batch in dataloader:
            feature_semantic_list = batch["semanticList"].to(get_device())

            feature_scene_offset = batch["scene_offset"].to(get_device())
            feature_motion = batch["motion"].to(get_device())
            feature_emotion = batch["emotion"].to(get_device())
            feature_loudness = batch["loudness"].to(get_device())
            feature_note_density = batch["note_density"].to(get_device())
            feature_instrument = batch["instrument"].to(get_device()).float()

            # Loudness_notedensity and Instrument
            ln_nd, inst = model(feature_semantic_list, 
                                feature_scene_offset,
                                feature_motion,
                                feature_emotion)
            
            ln_nd   = ln_nd.reshape(ln_nd.shape[0] * ln_nd.shape[1], -1)
            
            feature_loudness = feature_loudness.flatten().reshape(-1,1) # (batch_size, 300, 1)
            feature_note_density = feature_note_density.flatten().reshape(-1,1) # (batch_size, 300, 1)
            feature_combined = torch.cat((feature_note_density, feature_loudness), dim=1) # (batch_size, 300, 2)

            bce_instrument = F.binary_cross_entropy(inst, feature_instrument)
            sum_bce_instrument += float(bce_instrument)

            mse = torch.sqrt(F.mse_loss(ln_nd, feature_combined)) + bce_instrument
            sum_total_loss += float(mse)

            y_note_density, y_loudness = torch.split(ln_nd, split_size_or_sections=1, dim=1)

            rmse_note_density = torch.sqrt(F.mse_loss(y_note_density, feature_note_density))
            sum_rmse_note_density += float(rmse_note_density)
            
            rmse_loudness = torch.sqrt(F.mse_loss(y_loudness, feature_loudness))
            sum_rmse_loudness += float(rmse_loudness)
            
        avg_total_loss     = sum_total_loss / n_test
        avg_rmse_note_density     = sum_rmse_note_density / n_test
        avg_rmse_loudness     = sum_rmse_loudness / n_test
        avg_bce_instrument     = sum_bce_instrument / n_test

    return avg_total_loss, avg_rmse_note_density, avg_rmse_loudness, avg_bce_instrument
