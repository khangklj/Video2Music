from moviepy.editor import VideoFileClip
import os
from function import *
from tqdm import tqdm
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Download YouTube videos.')
    parser.add_argument('--format', type=str, default=1, help='Format of the idList.txt')
    parser.add_argument('--video_dir_path', type=str, default='./dataset/video', help='Path to save downloaded videos')
    return parser.parse_args()

def main(video_dir_path='./dataset/video', format=1):
    video_dir_path = video_dir_path
            
    # If the directory is not exist then create it
    if not os.path.exists(video_dir_path):
        os.makedirs(video_dir_path)

    idList = get_id_list(idlist_path='./dataset/vevo_meta/idlist.txt', format=format)

    # Download videos
    for index, id in tqdm(idList):
        video_path = os.path.join(video_dir_path, f'{index}.mp4')

        # If the video is not downloaded then download it
        if not os.path.exists(video_path):
            download_youtube_video(id, video_dir_path, f'{index}.mp4')

if __name__ == '__main__':
    args = parse_arguments()
    main(args.video_dir_path, args.format)