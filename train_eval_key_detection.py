from utilities.argument_reg_funcs import parse_eval_args, print_eval_args
from dataset.vevo_dataset import create_vevo_datasets
from model.video_regression import VideoRegression
from utilities.device import get_device, use_cuda
from utilities.constants import *

from tqdm import tqdm
import json as js
import joblib
import os
import numpy as np

# Support Vector Machines
from sklearn.svm import SVR

# Tree-Based Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Ensemble Models
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor

# Neural Networks
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, r2_score

split_ver = SPLIT_VER

def create_sample(sample, model, X, y):
    semantic = sample['semanticList'].unsqueeze(0)
    emotion = sample['emotion'].unsqueeze(0)
    feature = model.get_feature(semantic, None, None, emotion).squeeze().mean(dim=0)
    X.append(feature.detach().cpu().numpy())
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

    print('Create feature for key detection:')
    for sample in tqdm(train_dataset):
        create_sample(sample, model, X_train, y_train)

    for sample in tqdm(val_dataset):
        create_sample(sample, model, X_test, y_test)

    for sample in tqdm(test_dataset):
        create_sample(sample, model, X_test, y_test)

    print(f'Created {len(X_train)} training samples and {len(X_test)} testing samples')
    print(f'Each sample has shape ({X_train[0].shape}, {y_train[0].shape})')

    key_detection_models = {
        "SVR": SVR(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "RandomForestRegressor": RandomForestRegressor(),
        "AdaBoostRegressor": AdaBoostRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "BaggingRegressor": BaggingRegressor(),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "LinearRegression": LinearRegression(),
        "MLPRegressor": MLPRegressor(max_iter=500),
    }

    results = {}
    model_dir = 'saved_models/key_detection/'
    os.makedirs(model_dir, exist_ok=True)

    for name, model in key_detection_models.items():
        print(f"Training {name}...")
        
        model.fit(X_train, y_train)
        
        y_pred = np.round(model.predict(X_test))
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"MSE": mse, "R2": r2}
        print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

        model_path = os.path.join(model_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        print(f"Model {name} saved to {model_path}")

    print("\nSummary and save of Results:")
    for model_name, metrics in results.items():
        print(f"{model_name} - MSE: {metrics['MSE']:.4f}, R2: {metrics['R2']:.4f}")

    with open('key_detection_results.json', "w") as f:
        js.dump(results, f, indent=4)

if __name__ == "__main__":
    main()