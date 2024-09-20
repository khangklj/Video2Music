from moviepy.editor import VideoFileClip
import os
from function import *
from tqdm import tqdm

def main():
    video_dir_path = './dataset/video'
            
    # If the directory is not exist then create it
    if not os.path.exists(video_dir_path):
        os.makedirs(video_dir_path)

    idList = get_id_list(idlist_path='./dataset/vevo_meta/idlist.txt')

    # Download videos
    for index, id in tqdm(idList):
        video_path = os.path.join(video_dir_path, f'{index}.mp4')

        # If the video is not downloaded then download it
        if not os.path.exists(video_path):
            download_youtube_video(id, video_dir_path, f'{index}.mp4')

if __name__ == '__main__':
    main()