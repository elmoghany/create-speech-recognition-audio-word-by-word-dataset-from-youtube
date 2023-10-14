from pydub import AudioSegment
import os
import yt_dlp

def getAudio(youtube_url, audio_file_path, filename):
    print('********inside get audio********')

    download_path = os.path.join(audio_file_path, filename)
    if not os.path.exists(audio_file_path):
        os.makedirs(audio_file_path)
    print(f'download audio path: {download_path}')

    ydl_opts = {
        # 'ffmpeg-location': '.\\ffmpeg\\bin\\ffmpeg.exe',
        'outtmpl': f"{download_path}",  # Output format
        'prefer_ffmpeg': True,
        'keepvideo': True,
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }],
        'postprocessor_args': [
            '-ar', '16000',  # Set frame rate to 16000 Hz
            '-ac', '1',  # Set channels to 1 (mono)
        ],
    }

    video = yt_dlp.YoutubeDL(ydl_opts)
    try:
        video.download([youtube_url])
        print("Successfully Downloaded")
        print('setting the sampling rate')
        sound=AudioSegment.from_file(f'{download_path}'+'.wav')
        sound.set_frame_rate(16000)
        sound = sound.set_channels(1)
        sound = sound.set_sample_width(2)
        print('saving filtered audio')
        filtered_path = os.path.join(audio_file_path, filename + '_filtered.mp3')
        sound.export(filtered_path)
        os.remove(download_path)
        os.remove(f'{download_path}'+'.wav')
    except Exception as e:
        print("failed to Downloaded: ", e)
        
#     # Determine if an MP3 file contains speech based on energy thresholding
#     def is_speech(filtered_path=filtered_path, threshold=0.9, frame_length=2048, hop_length=512, high_energy_percentage=0.3):
        
#         # Load MP3 file into a numpy array
#         audio = AudioSegment.from_mp3(filtered_path)
#         samples = np.array(audio.get_array_of_samples())

#         # Convert stereo to mono if necessary
#         if audio.channels == 2:
#             print("2 channels, converting to mono")
#             samples = librosa.to_mono(samples.reshape(-1, 2).T)

#         # Convert integer samples to float and normalize
#         samples = samples.astype(np.float32) / (2**15)  # assuming 16-bit audio

#         # Extract the short-time energy of the audio
#         energy = np.abs(librosa.stft(samples, n_fft=frame_length, hop_length=hop_length))
#         energy = np.mean(energy**2, axis=0)
#         print("energy= ",energy)
        
#         # Check if a significant percentage of frames have energy above the threshold
#         high_energy_frames = np.sum(energy > threshold)
#         print("high_energy_frames= ",high_energy_frames)
        
#         print(f'high_energy_frames / float(len(energy)) = {high_energy_frames} / {float(len(energy))} = {high_energy_frames/float(len(energy))}')
#         print("low high_energy_percentage = speech = ",high_energy_percentage)

#         if high_energy_frames / float(len(energy)) > high_energy_percentage:
#             print(" high_energy_frames / float(len(energy)) > high_energy_percentage")
#             return True
#         else:
#             print(" high_energy_frames / float(len(energy)) < high_energy_percentage")
#             return False

#     isSpeech = is_speech(filtered_path)
#     print("is speech or not? ",isSpeech)

#     return isSpeech

# getAudio('https://youtube.com/watch?v=QS7RIooeYGE', './test-folder/', 'test')