from utilities.argument_reg_funcs import parse_eval_args, print_eval_args
from dataset.vevo_dataset import create_vevo_datasets
from model.video_regression import VideoRegression
from utilities.device import get_device, use_cuda
from utilities.constants import *

from tqdm import tqdm
import sklearn

split_ver = SPLIT_VER

def create_sample(sample, model, X, y):
    semantic = sample['semanticList'].unsqueeze(0)
    emotion = sample['emotion'].unsqueeze(0)
    feature = model.get_feature(semantic, None, None, emotion)
    X.append(feature.squeeze().detach().cpu().numpy())
    y.append(sample['key_val'])

def main():
    args = parse_eval_args()[0]

    train_dataset, val_dataset, test_dataset = create_vevo_datasets(
        dataset_root = "./dataset/", 
        max_seq_chord = args.max_sequence_chord, 
        max_seq_video = args.max_sequence_video, 
        vis_models = args.vis_models,
        emo_model = args.emo_model, 
        motion_type = args.motion_type,
        split_ver = SPLIT_VER, 
        random_seq = True,
        augmentation = args.augmentation)
    
    total_vf_dim = 0
    total_vf_dim += train_dataset[0]["semanticList"].shape[1]

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
    
    state_dict = torch.load(args.model_weights, map_location=get_device())
    model.load_state_dict(state_dict)

    X_train, y_train, X_test, y_test = [], [], [], []

    for sample in tqdm(train_dataset):
        create_sample(sample, model, X_train, y_train)

    for sample in tqdm(val_dataset):
        create_sample(sample, model, X_test, y_test)

    for sample in tqdm(test_dataset):
        create_sample(sample, model, X_test, y_test)

    print(f'Created {len(X_train)} training samples and {len(X_test)} testing samples')
    print(f'Each sample has shape ({X_train[0].shape}, {y_train[0].shape})')

if __name__ == "__main__":
    main()