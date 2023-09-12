#getAudio.py
import pytube
from pydub import AudioSegment
import os
#download libav for pydub

#Donwload audio
def getAudio(youtube_url, filename, audio_file_path):
    print('********inside get audio********')
    yt = pytube.YouTube(youtube_url)
    len(yt.streams)
    yt.bypass_age_gate()
    print('getting audio from youtube')
    stream = yt.streams.filter(only_audio=True)[0]
    download_path = os.path.join(audio_file_path, filename + '.mp3')
    if not os.path.exists(audio_file_path):
        os.makedirs(audio_file_path)
    print('download audio path: ', download_path)

    stream.download(filename=download_path)

    ####

    print('setting the sampling rate')
    sound=AudioSegment.from_file(download_path)
    sound.set_frame_rate(16000)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(2)
    print('saving filtered audio')
    filtered_path = os.path.join(audio_file_path, filename + '_filtered.mp3')
    sound.export(filtered_path)
