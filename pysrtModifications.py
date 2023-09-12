import pysrt
import os
from datetime import timedelta

def pysrtModifications(filename, subtitle_file_path):
    print('********inside  SRT Modifications********')

    download_subtitle_path = os.path.join(subtitle_file_path, filename)
    if not os.path.exists(subtitle_file_path):
        os.makedirs(subtitle_file_path)
    print('subtitle path: ', subtitle_file_path)

    subs = pysrt.open(download_subtitle_path+'_word_by_word.srt')

    for sub in subs:
        sub.end.milliseconds += 200 #adding 100 milliseconds to end time
        #handle the case of 999 milliseconds
        if sub.end.milliseconds >= 1000:
            sub.end.milliseconds -= 1000
            sub.end.seconds += 2


    subs.save(download_subtitle_path+'_word_by_word.srt', encoding='utf-8')
