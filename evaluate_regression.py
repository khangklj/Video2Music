import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.vevo_dataset import create_vevo_datasets

from model.video_regression import VideoRegression

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.argument_reg_funcs import parse_eval_args, print_eval_args

from utilities.run_model_regression import eval_model

import logging
import os
import sys

version = VERSION
split_ver = SPLIT_VER
split_path = "split_" + split_ver

VIS_MODELS_ARR = [
    "2d/clip_l14p"
]

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler('log/log_eval2.txt')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# main
def main( vm = "", isPrintArgs = True):
    args = parse_eval_args()[0]

    if isPrintArgs:
        print_eval_args(args)

    if vm != "":
        assert args.is_video == True
        args.vis_models = vm
        vis_arr = args.vis_models.split(" ")
        vis_arr.sort()
        vis_abbr_path = ""
        for v in vis_arr:
            vis_abbr_path = vis_abbr_path + "_" + VIS_ABBR_DIC[v]
        vis_abbr_path = vis_abbr_path[1:]
        args.model_weights = "./saved_models/" + version + "/best_rmse_weights.pickle"

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")
    
    _, _, test_dataset = create_vevo_datasets(
        dataset_root = "./dataset/", 
        max_seq_chord = args.max_sequence_chord, 
        max_seq_video = args.max_sequence_video, 
        vis_models = args.vis_models,
        emo_model = args.emo_model, 
        motion_type = args.motion_type,
        split_ver = SPLIT_VER, 
        random_seq = True,
        augmentation = args.augmentation)
    
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.n_workers)

    total_vf_dim = 0
    total_vf_dim += test_dataset[0]["semanticList"].shape[1]

    # total_vf_dim += 1 # Scene_offset

    # Motion
    # if args.motion_type == 0:
    #     total_vf_dim += 1 
    # elif args.motion_type == 1:
    #     total_vf_dim += 512
    # elif args.motion_type == 2:
    #     total_vf_dim += 768
    
    # Emotion
    if args.emo_model.startswith("6c"):
        total_vf_dim += 6
    else:
        total_vf_dim += 5
        
    model = VideoRegression(n_layers=args.n_layers, d_model=args.d_model, d_hidden=args.dim_feedforward, use_KAN=args.use_KAN, max_sequence_video=args.max_sequence_video, total_vf_dim=total_vf_dim, regModel=args.regModel).to(get_device())
    
    # Load weights
    state_dict = torch.load(args.model_weights, map_location=get_device())
    model.load_state_dict(state_dict)

    loss = nn.MSELoss()

    logging.info( f"VIS MODEL: {args.vis_models}" )
    logging.info("Evaluating (Note Density):")
    model.eval()
    
    eval_total_loss, eval_rmse_note_density, eval_rmse_loudness, eval_bce_instrument = eval_model(model, test_loader)

    logging.info(f"Avg Total loss: {eval_total_loss}")
    logging.info(f"Avg RMSE (Note Density): {eval_rmse_note_density}")
    logging.info(f"Avg RMSE (Loudness): {eval_rmse_loudness}")
    logging.info(f"Avg BCE (Instrument): {eval_bce_instrument}")

    logging.info(SEPERATOR)
    logging.info("")

if __name__ == "__main__":
    if len(VIS_MODELS_ARR) != 0 :
        for vm in VIS_MODELS_ARR:
            main(vm, False)
    else:
        main()
