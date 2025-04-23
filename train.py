import os
import csv
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, RAdam
from model.RAdanW import RAdanW
from lion_pytorch import Lion

from dataset.vevo_dataset import compute_vevo_accuracy, create_vevo_datasets

from model.music_transformer import MusicTransformer
from model.video_music_transformer import *

from model.loss import *

from utilities.constants import *
from utilities.device import get_device, use_cuda
from utilities.lr_scheduling import LrStepTracker, get_lr
from utilities.argument_funcs import parse_train_args, print_train_args, write_model_params

from utilities.run_model_vevo import train_epoch, eval_model
from third_party.log_maxvio import change_maxvio_logging_state
from utilities.constants import *

torch.autograd.set_detect_anomaly(True)

CSV_HEADER = ["Epoch", "Learn rate", 
              "Avg Train loss (total)", "Avg Train loss (chord)", "Avg Train loss (emotion)", 
              "Avg Train h1", "Avg Train h3", "Avg Train h5",
              "Avg Eval loss (total)", "Avg Eval loss (chord)", "Avg Eval loss (emotion)",
              "Avg Eval h1", "Avg Eval h3", "Avg Eval h5"]

BASELINE_EPOCH = -1
version = VERSION
split_ver = SPLIT_VER
split_path = "split_" + split_ver

VIS_MODELS_ARR = [
    "2d/clip_l14p"
]

# main
def main( vm = "" , isPrintArgs = True ):
    args = parse_train_args()[0]

    # Logging
    if args.music_gen_version != None and args.music_gen_version in ('2.3'):
        change_maxvio_logging_state(True)

    if isPrintArgs:
        print_train_args(args)
    if vm != "":
        args.vis_models = vm
    
    if args.is_video:
        vis_arr = args.vis_models.split(" ")
        vis_arr.sort()
        vis_abbr_path = ""
        for v in vis_arr:
            vis_abbr_path = vis_abbr_path + "_" + VIS_ABBR_DIC[v]
        vis_abbr_path = vis_abbr_path[1:]
    else:
        vis_abbr_path = "no_video"

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs( args.output_dir, exist_ok=True)
    os.makedirs( os.path.join( args.output_dir, version),  exist_ok=True)

    ##### Output prep #####
    params_file = os.path.join(args.output_dir, version, "model_params.txt")
    write_model_params(args, params_file)

    weights_folder = os.path.join(args.output_dir, version, "weights")
    os.makedirs(weights_folder, exist_ok=True)

    results_folder = os.path.join(args.output_dir, version)
    os.makedirs(results_folder, exist_ok=True)

    results_file = os.path.join(results_folder, "results.csv")
    best_loss_file = os.path.join(results_folder, "best_loss_weights.pickle")  
    best_text = os.path.join(results_folder, "best_epochs.txt")

    ##### Tensorboard #####
    if(args.no_tensorboard):
        tensorboard_summary = None
    else:
        from torch.utils.tensorboard import SummaryWriter
        tensorboad_dir = os.path.join(args.output_dir, version, "tensorboard")
        tensorboard_summary = SummaryWriter(log_dir=tensorboad_dir)
        
    train_dataset, val_dataset, test_dataset = create_vevo_datasets(
        dataset_root = "./dataset/", 
        max_seq_chord = args.max_sequence_chord, 
        max_seq_video = args.max_sequence_video, 
        vis_models = args.vis_models,
        motion_type = args.motion_type,
        emo_model = args.emo_model, 
        split_ver = SPLIT_VER, 
        random_seq = True, 
        is_video = args.is_video,
        augmentation = args.augmentation)
    
    total_vf_dim = 0

    if args.is_video:
        total_vf_dim += train_dataset[0]["semanticList"].shape[1]
        
        if args.scene_embed == False:
            total_vf_dim += 1 # Scene_offset
        
        # Motion
        if args.motion_type == 0:
            total_vf_dim += 1 
        elif args.motion_type == 1:
            total_vf_dim += 512
        elif args.motion_type == 2:
            total_vf_dim += 768
        
        # Emotion
        if args.emo_model.startswith("6c"):
            total_vf_dim += 6
        else:
            total_vf_dim += 5

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    train_loader_tmp = DataLoader(train_dataset, batch_size=1, num_workers=args.n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.n_workers)

    if args.music_gen_version == None:
        if args.is_video:
            model = VideoMusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                        d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                        max_sequence_midi=args.max_sequence_midi, max_sequence_video=args.max_sequence_video, 
                        max_sequence_chord=args.max_sequence_chord, total_vf_dim=total_vf_dim, rpr=args.rpr, 
                        scene_embed=args.scene_embed, chord_embed=args.chord_embed).to(get_device())
        else:
            model = MusicTransformer(n_layers=args.n_layers, num_heads=args.num_heads,
                        d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                        max_sequence_midi=args.max_sequence_midi, max_sequence_chord=args.max_sequence_chord, 
                        rpr=args.rpr).to(get_device())
    elif args.music_gen_version.startswith('1.'):
        model = VideoMusicTransformer_V1(version_name=args.music_gen_version, n_layers=args.n_layers, num_heads=args.num_heads,
                    d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                    max_sequence_midi=args.max_sequence_midi, max_sequence_video=args.max_sequence_video, 
                    max_sequence_chord=args.max_sequence_chord, total_vf_dim=total_vf_dim,
                    rms_norm=args.rms_norm, scene_embed=args.scene_embed, chord_embed=args.chord_embed,
                    dropTokenRate=args.droptoken).to(get_device())
    elif args.music_gen_version.startswith('2.'):
        model = VideoMusicTransformer_V2(version_name=args.music_gen_version, n_layers=args.n_layers, num_heads=args.num_heads,
                    d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                    max_sequence_midi=args.max_sequence_midi, max_sequence_video=args.max_sequence_video, 
                    max_sequence_chord=args.max_sequence_chord, total_vf_dim=total_vf_dim,
                    rms_norm=args.rms_norm, scene_embed=args.scene_embed, chord_embed=args.chord_embed,
                    dropTokenRate=args.droptoken, balancing=args.balancing).to(get_device())
    elif args.music_gen_version.startswith('3.'):
        model = VideoMusicTransformer_V3(version_name=args.music_gen_version, n_layers=args.n_layers, num_heads=args.num_heads,
                    d_model=args.d_model, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
                    max_sequence_midi=args.max_sequence_midi, max_sequence_video=args.max_sequence_video, 
                    max_sequence_chord=args.max_sequence_chord, total_vf_dim=total_vf_dim,
                    rms_norm=args.rms_norm, scene_embed=args.scene_embed, chord_embed=args.chord_embed,
                    dropTokenRate=args.droptoken).to(get_device())

    # Write model architecture
    with open(os.path.join(args.output_dir, 'model_architecture.txt'), 'w') as f:
        f.write(str(model))

    start_epoch = BASELINE_EPOCH
    if(args.continue_weights is not None):
        if(args.continue_epoch is None):
            print("ERROR: Need epoch number to continue from (-continue_epoch) when using continue_weights")
            assert(False)
        else:
            model.load_state_dict(torch.load(args.continue_weights, map_location=get_device()))
            start_epoch = args.continue_epoch
    elif(args.continue_epoch is not None):
        print("ERROR: Need continue weights (-continue_weights) when using continue_epoch")
        assert(False)

    ##### Lr Scheduler vs static lr #####
    if(args.lr is None):
        if(args.continue_epoch is None):
            init_step = 0
        else:
            init_step = args.continue_epoch * len(train_loader)
        lr = LR_DEFAULT_START
        lr_stepper = LrStepTracker(args.d_model, SCHEDULER_WARMUP_STEPS, init_step)
    else:
        lr = args.lr

    # Chord weights
    # chord_count = np.array([1 for _ in range(CHORD_SIZE)]) # Including PAD and EOS

    # for data in train_dataset:
    #     for item in data['chord']:
    #         chord_count[item] += 1

    # for data in val_dataset:
    #     for item in data['chord']:
    #         chord_count[item] += 1

    # for data in test_dataset:
    #     for item in data['chord']:
    #         chord_count[item] += 1
    
    # chord_weight = torch.tensor(1.0 / chord_count).float().to(get_device())
    # print(chord_weight)

    ##### SmoothCrossEntropyLoss or CrossEntropyLoss for training #####
    if args.ce_smoothing is None:
        train_loss_func = nn.CrossEntropyLoss(ignore_index=CHORD_PAD)
    elif args.ce_smoothing is not None:
        # FLAG
        # train_loss_func = SmoothCrossEntropyLoss(args.ce_smoothing, CHORD_SIZE, ignore_index=CHORD_PAD)
        if not args.auxiliary_loss:
            train_loss_func = nn.CrossEntropyLoss(ignore_index=CHORD_PAD, label_smoothing=args.ce_smoothing)
        else:
            train_loss_func = CombinedLoss([
                nn.CrossEntropyLoss(ignore_index=CHORD_PAD, label_smoothing=args.ce_smoothing),
                # nn.CrossEntropyLoss(weight=chord_weight, ignore_index=CHORD_PAD, label_smoothing=args.ce_smoothing),
                TopKAuxiliaryLoss(k=3, weight=3.0, vocab_size=CHORD_SIZE, ignore_index=CHORD_PAD),
                TopKAuxiliaryLoss(k=5, weight=5.0, vocab_size=CHORD_SIZE, ignore_index=CHORD_PAD)
            ], type='avg')

    eval_loss_func = train_loss_func

    eval_loss_emotion_func = nn.BCEWithLogitsLoss()
    train_loss_emotion_func = eval_loss_emotion_func

    ##### Optimizer #####
    if args.optimizer in (None, 'Adam'):
        opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
    elif args.optimizer == 'AdamW':
        opt = AdamW(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
    elif args.optimizer == 'RAdam':
        opt = RAdam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
    elif args.optimizer == 'RAdamW':
        opt = RAdam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON,
                    weight_decay=0.01, decoupled_weight_decay=True)
    elif args.optimizer == 'RAdanW':
        opt = RAdanW(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2, 0.92, 0.99), 
                     eps=ADAM_EPSILON, weight_decay=0.01)
    elif args.optimizer == 'Lion':
        opt = Lion(model.parameters(), lr=lr / 4, betas=(0.95, 0.98), weight_decay=1.0)

    if(args.lr is None):
        lr_scheduler = LambdaLR(opt, lr_stepper.step)
    else:
        lr_scheduler = None

    ##### Tracking best evaluation loss #####
    best_eval_loss       = float("inf")
    best_eval_loss_epoch = -1

    ##### Results reporting #####
    if(not os.path.isfile(results_file)):
        with open(results_file, "w", newline="") as o_stream:
            writer = csv.writer(o_stream)
            writer.writerow(CSV_HEADER)

    ##### TRAIN LOOP #####
    for epoch in range(start_epoch, args.epochs):
        if(epoch > BASELINE_EPOCH):
            print(SEPERATOR)
            print("NEW EPOCH:", epoch+1)
            print(SEPERATOR)
            print("")

            # Train
            train_epoch(epoch+1, model, train_loader, 
                        train_loss_func, train_loss_emotion_func,
                        opt, lr_scheduler, args.print_modulus, isVideo= args.is_video)

            print(SEPERATOR)
            print("Evaluating:")
        else:
            print(SEPERATOR)
            print("Baseline model evaluation (Epoch 0):")

        if not args.augmentation:
            train_metric_dict = eval_model(model, train_loader_tmp, 
                                    train_loss_func, train_loss_emotion_func,
                                    isVideo= args.is_video)
            
            train_total_loss = train_metric_dict["avg_total_loss"]
            train_loss_chord = train_metric_dict["avg_loss_chord"]
            train_loss_emotion = train_metric_dict["avg_loss_emotion"]
                
            train_h1 = train_metric_dict["avg_h1"]
            train_h3 = train_metric_dict["avg_h3"]
            train_h5 = train_metric_dict["avg_h5"]

        eval_metric_dict = eval_model(model, val_loader, 
                                eval_loss_func, eval_loss_emotion_func,
                                isVideo= args.is_video)
        
        eval_total_loss = eval_metric_dict["avg_total_loss"]
        eval_loss_chord = eval_metric_dict["avg_loss_chord"]
        eval_loss_emotion = eval_metric_dict["avg_loss_emotion"]
      
        eval_h1 = eval_metric_dict["avg_h1"]
        eval_h3 = eval_metric_dict["avg_h3"]
        eval_h5 = eval_metric_dict["avg_h5"]

        lr = get_lr(opt)

        print("Epoch:", epoch+1)

        if not args.augmentation:
            print("Avg train loss (total):", train_total_loss)
            print("Avg train loss (chord):", train_loss_chord)
            print("Avg train loss (emotion):", train_loss_emotion)

            print("Avg train h1:", train_h1)
            print("Avg train h3:", train_h3)
            print("Avg train h5:", train_h5)

        print("Avg val loss (total):", eval_total_loss)
        print("Avg val loss (chord):", eval_loss_chord)
        print("Avg val loss (emotion):", eval_loss_emotion)

        print("Avg val h1:", eval_h1)
        print("Avg val h3:", eval_h3)
        print("Avg val h5:", eval_h5)
        
        print(SEPERATOR)
        print("")

        new_best = False

        if(eval_total_loss < best_eval_loss):
            best_eval_loss       = eval_total_loss
            best_eval_loss_epoch = epoch+1
            torch.save(model.state_dict(), best_loss_file)
            new_best = True

        # Writing out new bests
        if(new_best):
            with open(best_text, "w") as o_stream:
                print("Best val loss epoch:", best_eval_loss_epoch, file=o_stream)
                print("Best val loss:", best_eval_loss, file=o_stream)
                
        if(not args.no_tensorboard):
            # tensorboard_summary.add_scalar("Avg_CE_loss/train", train_total_loss, global_step=epoch+1)
            tensorboard_summary.add_scalar("Avg_CE_loss/eval", eval_total_loss, global_step=epoch+1)
            # tensorboard_summary.add_scalar("Avg_CE_loss_chord/train", train_loss_chord, global_step=epoch+1)
            tensorboard_summary.add_scalar("Avg_CE_loss_chord/eval", eval_loss_chord, global_step=epoch+1)
            # tensorboard_summary.add_scalar("Avg_CE_loss_emotion/train", train_loss_emotion, global_step=epoch+1)
            tensorboard_summary.add_scalar("Avg_CE_loss_emotion/eval", eval_loss_emotion, global_step=epoch+1)
            tensorboard_summary.add_scalar("Learn_rate/train", lr, global_step=epoch+1)
            tensorboard_summary.flush()
            
        if((epoch+1) % args.weight_modulus == 0):
            epoch_str = str(epoch+1).zfill(PREPEND_ZEROS_WIDTH)
            path = os.path.join(weights_folder, "epoch_" + epoch_str + ".pickle")
            torch.save(model.state_dict(), path)
            
        with open(results_file, "a", newline="") as o_stream:
            writer = csv.writer(o_stream)
            if not args.augmentation:
                writer.writerow([epoch+1, lr, 
                                 train_total_loss, train_loss_chord, train_loss_emotion, train_h1, train_h3, train_h5,
                                eval_total_loss, eval_loss_chord, eval_loss_emotion, eval_h1, eval_h3, eval_h5])
            else:
                writer.writerow([epoch+1, lr, 
                                #  train_total_loss, train_loss_chord, train_loss_emotion, train_h1, train_h3, train_h5,
                                eval_total_loss, eval_loss_chord, eval_loss_emotion, eval_h1, eval_h3, eval_h5])
            
    # Sanity check just to make sure everything is gone
    if(not args.no_tensorboard):
        tensorboard_summary.flush()

    # Turn off logging
    change_maxvio_logging_state(False)

    return

if __name__ == "__main__":
    if len(VIS_MODELS_ARR) != 0 :
        for vm in VIS_MODELS_ARR:
            main(vm, False)
    else:
        main()
