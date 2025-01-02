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
from sklearn.svm import SVR, SVC

# Tree-Based Models
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Ensemble Models
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import BaggingRegressor, BaggingClassifier

# Neural Networks
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier

# Nearest Neighbors and Gaussian Naive Bayes
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

split_ver = SPLIT_VER

def create_sample(sample, model, X, y):
    semantic = sample['semanticList'].unsqueeze(0)
    emotion = sample['emotion'].unsqueeze(0)
    # feature = torch.cat((semantic, emotion), dim=-1).squeeze().mean(dim=0)
    # feature = model.get_feature(semantic, None, None, emotion).squeeze().mean(dim=0)
    # feature = model.get_feature(semantic, None, None, emotion).squeeze()[0, :]
    feature = model.get_feature(semantic, None, None, emotion).squeeze()[:5, :].flatten()

    emotion_idx = torch.argmax(emotion.mean(dim=0))

    if emotion_idx in (1, 2, 3): # Minor
        feature_key = torch.tensor([1]).float()
    else: # Major
        feature_key = torch.tensor([0]).float()

    feature = torch.cat((feature, feature_key))
    X.append(feature.detach().cpu().numpy())
    y.append(sample['key_val'].numpy())

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
        create_sample(sample, model, X_train, y_train)

    for sample in tqdm(test_dataset):
        create_sample(sample, model, X_test, y_test)

    print(f'Created {len(X_train)} training samples and {len(X_test)} testing samples')
    print(f'Each sample has shape ({X_train[0].shape}, {y_train[0].shape})')

    X_train = np.array(X_train)
    y_train = np.array(y_train).ravel()
    X_test = np.array(X_test)
    y_test = np.array(y_test).ravel()

    key_detection_models = {
        "SVR_linear": SVR(kernel='linear'),
        "SVR_poly": SVR(kernel='poly', degree=2),
        "SVR_rbf": SVR(kernel='rbf'),
        "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=50, max_features='sqrt', 
                                                       min_samples_leaf=5, min_samples_split=10),
        # "RandomForestRegressor_50": RandomForestRegressor(n_estimators=50,
        #                                                max_depth=20, 
        #                                                max_features='sqrt',
        #                                                min_samples_leaf=5,
        #                                                min_samples_split=10),
        # "RandomForestRegressor_100": RandomForestRegressor(n_estimators=100,
        #                                                max_depth=20, 
        #                                                max_features='sqrt',
        #                                                min_samples_leaf=5,
        #                                                min_samples_split=10),
        "RandomForestRegressor_150": RandomForestRegressor(n_estimators=150,
                                                       max_depth=10, 
                                                       max_features='sqrt',
                                                       min_samples_leaf=5,
                                                       min_samples_split=10),
        # "RandomForestRegressor_200": RandomForestRegressor(n_estimators=200,
        #                                                max_depth=20, 
        #                                                max_features='sqrt',
        #                                                min_samples_leaf=5,
        #                                                min_samples_split=10),
        # "RandomForestRegressor_250": RandomForestRegressor(n_estimators=250,
        #                                                max_depth=20,
        #                                                max_features='sqrt',
        #                                                min_samples_leaf=5,
        #                                                min_samples_split=10),
        "AdaBoostRegressor": AdaBoostRegressor(loss='square'),
        "GradientBoostingRegressor": GradientBoostingRegressor(loss='huber', warm_start=True, min_samples_leaf=5, 
                                                               min_samples_split=10, max_features='sqrt'),
        "BaggingRegressor": BaggingRegressor(max_samples=5, max_features=5, warm_start=True),
        # "KNeighborsRegressor_3": KNeighborsRegressor(n_neighbors=3),
        # "KNeighborsRegressor_5": KNeighborsRegressor(n_neighbors=5),
        # "KNeighborsRegressor_9": KNeighborsRegressor(n_neighbors=9),
        # "KNeighborsRegressor_15": KNeighborsRegressor(n_neighbors=15),
        # "LinearRegression": LinearRegression(),
        "MLPRegressor": MLPRegressor(hidden_layer_sizes=256, solver='lbfgs', 
                                     learning_rate='adaptive', max_iter=200),
    }

    # key_detection_models = {
    #     'SVC': SVC(),
    #     'DecisionTreeClassifier': DecisionTreeClassifier(),
    #     'RandomForestClassifier_50': RandomForestClassifier(n_estimators=50, max_depth=20, max_features='sqrt', min_samples_leaf=5, min_samples_split=10),
    #     'RandomForestClassifier_100': RandomForestClassifier(n_estimators=100, max_depth=20, max_features='sqrt', min_samples_leaf=5, min_samples_split=10),
    #     'RandomForestClassifier_150': RandomForestClassifier(n_estimators=150, max_depth=20, max_features='sqrt', min_samples_leaf=5, min_samples_split=10),
    #     'RandomForestClassifier_200': RandomForestClassifier(n_estimators=200, max_depth=20, max_features='sqrt', min_samples_leaf=5, min_samples_split=10),
    #     'RandomForestClassifier_250': RandomForestClassifier(n_estimators=250, max_depth=20, max_features='sqrt', min_samples_leaf=5, min_samples_split=10),
    #     'AdaBoostClassifier': AdaBoostClassifier(),
    #     'GradientBoostingClassifier': GradientBoostingClassifier(),
    #     'BaggingClassifier': BaggingClassifier(),
    #     'KNeighborsClassifier_3': KNeighborsClassifier(n_neighbors=3),
    #     'KNeighborsClassifier_5': KNeighborsClassifier(n_neighbors=5),
    #     'KNeighborsClassifier_9': KNeighborsClassifier(n_neighbors=9),
    #     'KNeighborsClassifier_15': KNeighborsClassifier(n_neighbors=15),
    #     'MLPClassifier': MLPClassifier(max_iter=500),
    #     'GaussianNB': GaussianNB(),
    # }

    results = {}
    model_dir = 'saved_models/key_detection/'
    os.makedirs(model_dir, exist_ok=True)

    n_show = 22
    for name, model in key_detection_models.items():
        print(f"Training {name}... ===================")
        
        model.fit(X_train, y_train)

        ### Training predictions
        y_pred = np.round(model.predict(X_train))

        mse = mean_squared_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        
        results[name] = {"MSE": mse, "R2": r2}
        print(f"Train {name} - MSE: {mse:.4f}, R2: {r2:.4f}")

        print('Training predictions')
        print(y_pred[:n_show].astype(int))
        print(y_train[:n_show])

        # y_pred = model.predict(X_train)

        # acc = accuracy_score(y_pred, y_train)
        # f1 = f1_score(y_pred, y_train, average='weighted')

        # results[name] = {"Acc": acc, "F1": f1}
        # print(f"Train {name} - Acc: {acc:.4f}, F1: {f1:.4f}")

        # print('Training predictions')
        # print(y_pred[:n_show])
        # print(y_train[:n_show])
        
        ### Testing predictions
        y_pred = np.round(model.predict(X_test))
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {"MSE": mse, "R2": r2}
        print(f"Test {name} - MSE: {mse:.4f}, R2: {r2:.4f}")

        print('Testing predictions')
        print(y_pred[:n_show].astype(int))
        print(y_test[:n_show])

        # y_pred = model.predict(X_test)
        
        # acc = accuracy_score(y_pred, y_test)
        # f1 = f1_score(y_pred, y_test, average='weighted')
        
        # results[name] = {"Acc": acc, "F1": f1}
        # print(f"Test {name} - Acc: {acc:.4f}, F1: {f1:.4f}")

        # print('Testing predictions')
        # print(y_pred[:n_show])
        # print(y_test[:n_show])
        
        ### Save the trained model
        model_path = os.path.join(model_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        # print(f"Model {name} saved to {model_path}")

    print("\nSummary and save of Results:")
    for model_name, metrics in results.items():
        print(f"{model_name} - MSE: {metrics['MSE']:.4f}, R2: {metrics['R2']:.4f}")
        # print(f"{model_name} - Acc: {metrics['Acc']:.4f}, F1: {metrics['F1']:.4f}")

    with open('key_detection_results.json', "w") as f:
        js.dump(results, f, indent=4)

if __name__ == "__main__":
    main()