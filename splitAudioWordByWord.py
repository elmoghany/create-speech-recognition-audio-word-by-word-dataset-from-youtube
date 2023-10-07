from pydub import AudioSegment
import re
import os 
import csv
from skipWords import WORDS_TO_SKIP

def splitAudioWordByWord(audio_file_path, subtitle_file_path, whole_filename, output_file_path, output_file_name):
    print('********inside  split audio word by word********')

    skip_path = os.path.join(subtitle_file_path, 'skip.xml')
    if (os.path.isfile(skip_path)):
        print("No splits are done - skip")

    else:
        def parse_srt(download_subtitle_path):
            with open(download_subtitle_path, 'r', encoding='utf-8') as f:
                srt_content = f.read()

            pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?:\n{2}|\Z)', re.DOTALL)
            parsed_srt = []

            for m in re.finditer(pattern, srt_content):
                start_time_str = m.group(2).replace(',', '.')
                end_time_str = m.group(3).replace(',', '.')
                
                start_time = sum(x * int(t) for x, t in zip([3600000, 60000, 1000, 1], re.split('[:.]', start_time_str)))
                end_time = sum(x * int(t) for x, t in zip([3600000, 60000, 1000, 1], re.split('[:.]', end_time_str)))
                
                parsed_srt.append({
                    'index': m.group(1),
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': m.group(4).strip()
                })
            
            return parsed_srt

        print('original audio    path: ', audio_file_path)
        print('original subtitle path: ', subtitle_file_path)

        download_audio_path = os.path.join(audio_file_path, whole_filename)
        if not os.path.exists(audio_file_path):
            os.makedirs(audio_file_path)
        download_subtitle_path = os.path.join(subtitle_file_path, whole_filename)
        if not os.path.exists(subtitle_file_path):
            os.makedirs(subtitle_file_path)

        print('download audio    path: ', download_audio_path)
        print('download subtitle path: ', download_subtitle_path)

        audio = AudioSegment.from_file(download_audio_path+'_filtered.mp3', format="mp3")
        parsed_srt = parse_srt(download_subtitle_path+'_word_by_word.srt')


        rows = []  # Initialize an empty list to hold rows

        print('original output path: ', output_file_path)

        download_output_path = os.path.join(output_file_path, output_file_name)
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)
        print('download output path: ', download_output_path)

        for entry in parsed_srt:
            start_time = entry['start_time']
            end_time = entry['end_time']
            index = entry['index']
            text = entry['text']
            if text.lower() in WORDS_TO_SKIP:
                # print(f"Skipping word: {text}")
                continue

            segment = audio[start_time:end_time]

            # Construct the segment's save path
            # print(f'processing {index}_{text[:10]}.mp3')
            segment_file_path = os.path.join(download_output_path, f"{index}_{text[:10]}.mp3")

            if not os.path.exists(os.path.dirname(segment_file_path)):
                os.makedirs(os.path.dirname(segment_file_path))
            # print('segment file path: ', segment_file_path)

            segment.export(segment_file_path, format="mp3")
            
            row = {'path': segment_file_path, 'transcription': text}
            rows.append(row)  # Append row dictionary to list

        with open(output_file_path+'metadata.csv', 'w', newline='') as csvfile:
            fieldnames = ['path', 'transcription']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write the header
            writer.writeheader()

            # Write the rows
            writer.writerows(rows)
