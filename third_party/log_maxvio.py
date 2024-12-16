import torch
import numpy as np
import os

is_logging = True
c_patch = None

def change_maxvio_logging_state(state):
    global is_logging
    is_logging = state

def update_maxvio(c):
    global is_logging
    global c_patch
    
    if not is_logging:
        return

    if c_patch is None:
        c_patch = torch.zeros_like(c)
    c_patch += c

def reset_maxvio():
    global is_logging    
    global c_patch

    if not is_logging:
        return

    c_patch = None

def save_maxvio():
    global is_logging
    global c_patch
    
    if not is_logging or c_patch is None:
        return
    
    if not os.path.exists("log/"):
        os.makedirs("log/")

    # Calculate maxvio
    load = c_patch
    load_mean = torch.mean(load)
    max_vio = torch.max(load - load_mean) / load_mean

    max_vio = max_vio.cpu().numpy()

    if os.path.exists("log/maxvio.npy"):        
        arr = np.load("log/maxvio.npy")        
        arr = np.hstack((arr, max_vio))
    else:
        arr = np.array([max_vio])
        
    np.save("log/maxvio.npy", arr)
        