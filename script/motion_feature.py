import os
import math
import cv2
import numpy as np
import torchvision.models as models
from tqdm import tqdm
import torch
from PIL import Image

TORCH_CPU_DEVICE = torch.device("cpu")

if(torch.cuda.device_count() > 0):
    TORCH_CUDA_DEVICE = torch.device("cuda:0")
else:
    print("----- WARNING: CUDA devices not detected. This will cause the model to run very slow! -----")
    print("")
    TORCH_CUDA_DEVICE = None

USE_CUDA = True

# get_device
def get_device():
    """
    ----------
    Author: Damon Gwinn
    ----------
    Grabs the default device. Default device is CUDA if available and use_cuda is not False, CPU otherwise.
    ----------
    """

    if((not USE_CUDA) or (TORCH_CUDA_DEVICE is None)):
        return TORCH_CPU_DEVICE
    else:
        return TORCH_CUDA_DEVICE

def main():
    directory = "../dataset/vevo_chord/lab/all/"
    directory_vevo = "../dataset/vevo/"
    datadict = {}

    model = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT)   
    model.classifier = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten()
    )
    model = model.to(get_device())
    model.eval()
    transform = models.MaxVit_T_Weights.IMAGENET1K_V1.transforms()

    for filename in sorted(os.listdir(directory_vevo)):
        print(filename, end='\t')
        fname = filename.split(".")[0]
        videopath = os.path.join(directory_vevo, filename.replace("lab", "mp4"))
        cap = cv2.VideoCapture(videopath)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        time_incr = 1 / fps
        prev_frame = None
        prev_time = 0
        motion_value = 0
        sec = 0
        motiondict = {}

        # while cap.isOpened():
        #     # Read the frame and get its time stamp
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #     curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
        #     # Calculate the RGB difference between consecutive frames per second
        #     motiondict[0] = "0.0000"

        #     if prev_frame is not None and curr_time - prev_time >= 1:
        #         diff = cv2.absdiff(frame, prev_frame)
        #         diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)

        #         motion_value = diff_rgb.mean()
        #         #print("Motion value for second {}: {}".format(int(curr_time), motion_value))
        #         motion_value = format(motion_value, ".4f")
        #         motiondict[int(curr_time)] = str(motion_value)
        #         prev_time = int(curr_time)

        #     # Update the variables
        #     prev_frame = frame.copy()
        # # Release the video file and close all windows
        # cap.release()
        # cv2.destroyAllWindows()

        # fpathname = "../dataset/vevo_motion/all/" + fname + ".lab"
        # with open(fpathname,'w',encoding = 'utf-8') as f:
        #     for i in range(0, len(motiondict)):
        #         f.write(str(i) + " "+motiondict[i]+"\n")

        # === Our option 1 === #
        features = []
        while cap.isOpened():
            # Read the frame and get its time stamp
            ret, frame = cap.read()
            if not ret:
                break
            curr_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Calculate the RGB difference between consecutive frames per second
            if prev_frame is not None and curr_time - prev_time >= 1:
                diff = cv2.absdiff(frame, prev_frame)
                diff_rgb = cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)

                diff_image = transform(Image.fromarray(diff_rgb)).unsqueeze(0).to(get_device())
                with torch.no_grad():
                    motion_features = model(diff_image).squeeze()

                # print("Motion value for second {}: {}".format(int(curr_time), motion_features.shape))
                motion_features = motion_features.cpu().numpy()
                features.append(motion_features)

            # Update the variables
            prev_frame = frame.copy()
        # Release the video file and close all windows
        cap.release()
        cv2.destroyAllWindows()

        features = np.stack(features, axis=0)
        print(features.shape)
        np.save("../dataset/vevo_motion/option1/" + fname + ".npy", features)

if __name__ == "__main__":
    main()

