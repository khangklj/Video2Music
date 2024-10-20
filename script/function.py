'''
================================================================================================
            THIS IS PROTOTYPE

    def download_youtube_video(video_id, save_path=None, new_filename=None) -> None
    def get_id_list(idlist_path='idlist.txt') -> List[(str, str)]
    def get_frame_list(video, step=1) -> List[np.array(shape=(x, y, 3))]
    def get_video_audio(video_path) -> (VideoFileClip, AudioFileClip)
'''

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
================================================================================================
This function is used to download Youtube video from video's id
Parameter:
    video_id: id of Youtube video
    save_path: path to save the video (default is current directory)
    new_filename: name of the downloaded video (default is the name that Youtube provide)
Return: None
'''
import os
from pytubefix import YouTube

def download_youtube_video(video_id, save_path=None, new_filename=None):
    try:
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        yt = YouTube(video_url)
        stream = yt.streams.get_highest_resolution()

        # Get current path if save path is not provided
        if not save_path:
            save_path = os.getcwd()

        # Download the video
        download_path = stream.download(output_path=save_path)
        print(f'Download {video_url} completed')
        
        # Rename if a new filename is provided
        if new_filename:
            new_file_path = os.path.join(save_path, new_filename)
            os.rename(download_path, new_file_path)

    except Exception as e:
        print(f'An error occurred: {e}, video url: {video_url}')


'''
================================================================================================
This function is used to get video's id
Parameter:
    idlist_path: path of file idlist.txt
Return: List[(str, str)] # (index, video_id)
'''
def get_id_list(idlist_path='idlist.txt', format=1):
    ans = []
    with open(idlist_path, 'r', encoding='utf-8') as fi:
        lines = fi.readlines()

    for line in lines[:]:
        tmp = line.split('\t')
        if format == 1:
          index, id = tmp[0][:3], tmp[1][:-1]
        elif format == 2:
          index, id = tmp[0], tmp[1][:-1]
        ans.append((index, id))

    return ans


'''
================================================================================================
This function is used get the frame list of the video by fps
Parameter:
    video: moviepy.video.io.VideoFileClip.VideoFileClip object to get the frame
    fps: the number frames each second (default is 1 frame per second)
Return: List of frames by fps
'''
from moviepy.editor import VideoFileClip

def get_frame_list(video: VideoFileClip, fps=1):
    return list(video.iter_frames(fps))

'''
================================================================================================
This function is used to seperate video and audio from video file
Parameter:
    video_path: video file name to get video and audio (include path)
Return: (VideoFileClip object, AudioFileClip object) both from moviepy
'''
from moviepy.editor import VideoFileClip

def get_video_audio(video_path):
    video = VideoFileClip(video_path)
    return video.without_audio(), video.audio