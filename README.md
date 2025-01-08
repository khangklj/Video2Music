# Video2Music: Suitable Music Generation from Videos using an Affective Multimodal Transformer model

<div align="center">
  <img src="v2m.png" width="400"/>
</div>

## Introduction
We propose a novel AI-powered multimodal music generation framework called Video2Music. This framework uniquely uses video features as conditioning input to generate matching music using a Transformer architecture. By employing cutting-edge technology, our system aims to provide video creators with a seamless and efficient solution for generating tailor-made background music.

## Quickstart Guide

Generate music from video:

```python
import IPython
from video2music import Video2music

input_video = "input.mp4"

input_primer = "C Am F G"
input_key = "C major"

video2music = Video2music()
output_filename = video2music.generate(input_video, primer=input_primer, key=input_key)

IPython.display.Video(output_filename)
```

## Installation

**For training**

```bash
git clone https://github.com/Phan-Trung-Thuan/Video2Music.git
cd Video2Music
pip install -r requirements.txt
pip install --upgrade gensim
```

**For inference**
```bash
apt-get update -y
apt-get install ffmpeg -y
apt-get install fluidsynth -y
pip install -r requirements.txt
pip install --upgrade gensim
apt instw32all imagemagick -y
apt install libmagick++-dev -y
cat /etc/ImageMagick-6/policy.xml | sed 's/none/read,write/g'> /etc/ImageMagick-6/policy.xml
```

* Download the default soundfont file [default_sound_font.sf2](https://drive.google.com/file/d/1B9qjgimW9h6Gg5k8PZNt_ArWwSMJ4WuJ/view?usp=drive_link) or using custom soundfonts processed by us [soundfonts.zip](https://drive.google.com/uc?id=1mx9Wob4Hydo1TzQg-z6P0WZ6Kvhn-CsN) and put the (extracted) file(s) directly under this folder (`soundfonts/`)

  Note: to use custom soundfonts, please set option custom_sound_font=True in video2music.generate() (video2music.py) 

* Our code is built on pytorch version 1.12.1 (torch==1.12.1 in the requirements.txt). But you might need to choose the correct version of `torch` based on your CUDA version

## Dataset

* Obtain the dataset:
  * MuVi-Sync-dataset-v3 [(Link)](https://kaggle.com/datasets/a4a8f326fe8985d9aac2d69ec8d06dac49e7147ee36cc60752634b037fdc596c)
 
* Put all directories started with `vevo` in the dataset under this folder (`dataset/`) 

## Directory Structure

* `saved_models/`: saved model files
* `utilities/`
  * `run_model_vevo.py`: code for running model (AMT)
  * `run_model_regression.py`: code for running regression model
  * `argument_funcs.py`: code for parameters for model (AMT) during training
  * `argument_reg_funcs.py`: code for parameters for regression model during training
  * `argument_generate_funcs.py`: code for parameters for both model during inference
* `model/`
  * `video_music_transformer.py`: Affective Multimodal Transformer (AMT) model 
  * `video_regression.py`: Regression model used for predicting note density/loudness
* `dataset/`
  * `vevo_dataset.py`: Dataset loader
* `script/` : code for extracting video/music features (sementic, motion, emotion, scene offset, loudness, and note density)
* `train.py`: training script (AMT)
* `train_regression.py`: training script (regression model)
* `evaluate.py`: evaluation script
* `generate.py`: inference script
* `video2music.py`: Video2Music module that outputs video with generated background music from input video

## Training

  ```shell
  python train.py
  ```

  or
  
  ```shell
  python train_regression.py
  ```

## Inference

  ```shell
  python generate.py
  ```

## TODO

- ...

## Citation
If you find this resource useful, [please cite the original work](https://doi.org/10.1016/j.eswa.2024.123640):

```bibtex
@article{KANG2024123640,
  title = {Video2Music: Suitable music generation from videos using an Affective Multimodal Transformer model},
  author = {Jaeyong Kang and Soujanya Poria and Dorien Herremans},
  journal = {Expert Systems with Applications},
  pages = {123640},
  year = {2024},
  issn = {0957-4174},
  doi = {https://doi.org/10.1016/j.eswa.2024.123640},
}
```

Kang, J., Poria, S. & Herremans, D. (2024). Video2Music: Suitable Music Generation from Videos using an Affective Multimodal Transformer model, Expert Systems with Applications (in press).


## Acknowledgements

Our code is based on [Music Transformer](https://github.com/gwinndr/MusicTransformer-Pytorch).


