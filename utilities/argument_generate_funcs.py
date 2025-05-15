import argparse
from .constants import *

version = VERSION
split_ver = SPLIT_VER
split_path = "split_" + split_ver

def print_eval_args(args):
    print(SEPERATOR)
    print("input_dir_music:", args.input_dir_music)
    print("input_dir_video:", args.input_dir_video)

    print("model_weights:", args.model_weights)
    print("n_workers:", args.n_workers)
    print("force_cpu:", args.force_cpu)
    print("")
    print("batch_size:", args.batch_size)
    print("")
    print("rpr:", args.rpr)
    
    print("max_sequence_midi:", args.max_sequence_midi)
    print("max_sequence_video:", args.max_sequence_video)
    print("max_sequence_chord:", args.max_sequence_chord)
    
    print("n_layers:", args.n_layers)
    print("num_heads:", args.num_heads)
    print("d_model:", args.d_model)
    print("")
    print("dim_feedforward:", args.dim_feedforward)
    print(SEPERATOR)
    print("")

# parse_generate_args
def parse_generate_args():
    parser = argparse.ArgumentParser()
    outputpath = "./output_vevo/" + version
    if IS_VIDEO:
        modelpath = "saved_models/AMT/best_loss_weights.pickle"
        modelpathReg = "saved_models/AMT/best_rmse_weights.pickle"
        modelpathKey = "saved_models/AMT/SVC_rbf.pkl"
        # modelpath = "./saved_models/"+version+ "/"+VIS_MODELS_PATH+"/results/best_acc_weights.pickle"
        # modelpathReg = "./saved_models/"+version+ "/"+VIS_MODELS_PATH+"/results_regression_bigru/best_rmse_weights.pickle"
    else:
        modelpath = "./saved_models/" + version + "/no_video/results/best_loss_weights.pickle"
        modelpathReg = None
        modelpathKey = "./saved_models/" + version + "/no_video/results/SVC_rbf.pkl"

    parser.add_argument("-dataset_dir", type=str, default="./dataset/", help="Folder of VEVO dataset")
    
    parser.add_argument("-input_dir_music", type=str, default="./dataset/vevo_chord/" + MUSIC_TYPE, help="Folder of video CNN feature files")
    parser.add_argument("-input_dir_video", type=str, default="./dataset/vevo_vis", help="Folder of video CNN feature files")

    parser.add_argument("-output_dir", type=str, default= outputpath, help="Folder to write generated midi to")

    parser.add_argument("-primer_file", type=str, default=None, help="File path or integer index to the evaluation dataset. Default is to select a random index.")
    parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")

    parser.add_argument("-target_seq_length_midi", type=int, default=1024, help="Target length you'd like the midi to be")
    parser.add_argument("-target_seq_length_chord", type=int, default=300, help="Target length you'd like the midi to be")
    
    parser.add_argument("-num_prime_midi", type=int, default=256, help="Amount of messages to prime the generator with")
    parser.add_argument("-num_prime_chord", type=int, default=30, help="Amount of messages to prime the generator with")    
    parser.add_argument("-model_weights", type=str, default=modelpath, help="Pickled model weights file saved with torch.save and model.state_dict()")
    parser.add_argument("-modelReg_weights", type=str, default=modelpathReg, help="Pickled model weights file saved with torch.save and model.state_dict()")
    parser.add_argument("-modelKey_weights", type=str, default=modelpathKey, help="Pickled model weights file saved with torch.save and model.state_dict()")
    

    parser.add_argument("-beam", type=int, default=0, help="Beam search k. 0 for random probability sample and 1 for greedy")

    parser.add_argument("-max_sequence_midi", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("-max_sequence_video", type=int, default=300, help="Maximum video sequence to consider")
    parser.add_argument("-max_sequence_chord", type=int, default=300, help="Maximum chord sequence to consider")

    parser.add_argument("-chord_embed", type=bool, default=False, help="Use chord embedding or not")
    
    # Chord generation model
    parser.add_argument("-n_layers", type=int, default=6, help="Number of decoder layers to use")
    parser.add_argument("-num_heads", type=int, default=8, help="Number of heads to use for multi-head attention")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")
    parser.add_argument("-dim_feedforward", type=int, default=1024, help="Dimension of the feedforward layer")
    parser.add_argument('-rms_norm', type=bool, default=False, help="Use RMSNorm instead of LayerNorm")
    parser.add_argument('-music_gen_version', type=str, default=None, help="Version number. None is original musgic generation AMT model")
    parser.add_argument("-scene_embed", type=bool, default=False, help="Use scene offset embedding or not")
    parser.add_argument("-balancing", type=bool, default=False, help="False / True")

    # Reg model
    parser.add_argument("-n_layers_reg", type=int, default=2, help="Number of layers to use")
    parser.add_argument("-d_model_reg", type=int, default=64, help="Dimension of the model (output dim of embedding layers, etc.)")
    parser.add_argument("-dim_feedforward_reg", type=int, default=256, help="Dimension of the feedforward layer")
    parser.add_argument('-use_KAN_reg', type=bool, default=False, help="Use KANLinear instead of Linear")
    parser.add_argument('-regModel', type=str, default='bigru', help="Version name. None is original loudness and note density Regression model")

    parser.add_argument("-is_video", type=bool, default=IS_VIDEO, help="MusicTransformer or VideoMusicTransformer")

    if IS_VIDEO:
        parser.add_argument("-vis_models", type=str, default=VIS_MODELS_SORTED, help="...")
    else:
        parser.add_argument("-vis_models", type=str, default="", help="...")

    parser.add_argument("-emo_model", type=str, default="6c_l14p", help="...")
    parser.add_argument("-motion_type", type=int, default=0, help="0 as original, 1 as our option 1, 2 as out option 2")
    parser.add_argument("-rpr", type=bool, default=RPR, help="...")

    return parser.parse_known_args()

def print_generate_args(args):
    
    print(SEPERATOR)
    print("input_dir_music:", args.input_dir_music)
    print("input_dir_video:", args.input_dir_video)

    print("output_dir:", args.output_dir)
    print("primer_file:", args.primer_file)
    print("force_cpu:", args.force_cpu)
    print("")

    print("target_seq_length_midi:", args.target_seq_length_midi)
    print("target_seq_length_chord:", args.target_seq_length_chord)
    
    print("num_prime_midi:", args.num_prime_midi)
    print("num_prime_chord:", args.num_prime_chord)

    print("model_weights:", args.model_weights)
    print("modelReg_weights: ", args.modelReg_weights)
    print("beam:", args.beam)

    print("")
    print("rpr:", args.rpr)    
    print("max_sequence_midi:", args.max_sequence_midi)
    print("max_sequence_video:", args.max_sequence_video)
    print("max_sequence_chord:", args.max_sequence_chord)
    
    
    print("")
    print("CHORD GENERATION MODEL")
    print("music_gen_version: ", args.music_gen_version)
    print("n_layers:", args.n_layers)
    print("num_heads:", args.num_heads)
    print("d_model:", args.d_model)
    print("dim_feedforward:", args.dim_feedforward)
    print("rms_norm: ", args.rms_norm)
    print("vis_models: ", args.vis_models)
    print("emo_model: ", args.emo_model)
    print("motion_type: ", args.motion_type)

    print("")
    print("REGRESSION MODEL")
    print("regModel: ", args.regModel)
    print("n_layers_reg:", args.n_layers_reg)
    print("d_model_reg:", args.d_model_reg)
    print("dim_feedforward_reg:", args.dim_feedforward_reg)
    print("use_KAN_reg: ", args.use_KAN_reg)
    
    print("")

    print(SEPERATOR)
    print("")
