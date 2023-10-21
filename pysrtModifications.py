import pysrt
import os
from datetime import timedelta

def pysrtModifications(subtitle_file_path,filename):
    print('********4) Inside  SRT Modifications********')
    skip_path = os.path.join(subtitle_file_path, 'skip.xml')
    if (os.path.isfile(skip_path)):
        print("No modification to any subtitle  - skip")
    else:
        download_subtitle_path = os.path.join(subtitle_file_path, filename)
        if not os.path.exists(subtitle_file_path):
            os.makedirs(subtitle_file_path)
        # print('subtitle path: ', subtitle_file_path)

        subs = pysrt.open(download_subtitle_path+'_word_by_word.srt', encoding='iso-8859-1')

        for sub in subs:
            sub.end.milliseconds += 200 #adding 200 milliseconds to end time
            #handle the case of 999 milliseconds
            if sub.end.milliseconds >= 1000:
                sub.end.milliseconds -= 1000
                sub.end.seconds += 2


        subs.save(download_subtitle_path+'_word_by_word.srt', encoding='utf-8')
