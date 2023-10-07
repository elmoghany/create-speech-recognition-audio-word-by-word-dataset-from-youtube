#main.py
import heapq
import itertools
from getAudio import getAudio
from getXMLSubtitle import getXMLSubtitle 
from xMLToWordByWordSRT import xMLToWordByWordSRT
from splitAudioWordByWord import splitAudioWordByWord
from pysrtModifications import pysrtModifications
from modelFineTuning import modelFineTuning
from getChannelVideoUrls import getChannelVideoUrls
import ijson
import json


def main():
    channelNames=[  
                    'https://www.youtube.com/@Tsaog'
                    ,'https://www.youtube.com/@goldenstateortho918'
                    ,'https://www.youtube.com/@orthopaedicacademy'
                    ,'https://www.youtube.com/@ApurvMehradr'
                ]

    for channel_url in channelNames:
        videoUrls=[]
        videoTitles=[]
        channel_name = channel_url.split('/@')[-1]  # Extracts the part after the last slash
        print("******************************PROCESSING CHANNELS*********************************")
        print("******************************PROCESSING CHANNELS*********************************")
        print("******************************PROCESSING CHANNELS*********************************")
        print("******************************PROCESSING CHANNELS*********************************")
        print("******************************PROCESSING CHANNELS*********************************")
        print(f"Processing Channel: {channel_name}")

        videoTitles, videoUrls = getChannelVideoUrls(channel_url)
        print("length of videoTitles",len(videoTitles))
        print("length of videoUrls",len(videoUrls))

        number_of_videos = (len(videoUrls))
        for i in range(len(videoUrls)):
            print("****************************PROCESSING VIDEO in CHANNEL*********************************")
            videoTitle = videoTitles.pop()
            videoUrl = videoUrls.pop()
            wholeFileName="whole"
            channelPath=f'./dataset/{channel_name}/'
            channelSkippedVideos=f'./dataset/{channel_name}/{channel_name}'
            wholeSubtitlePath=f'./dataset/{channel_name}/{videoTitle}/wholeSubtitle/'
            wholeAudioPath=f'./dataset/{channel_name}/{videoTitle}/wholeAudio/'
            processedAudioPath=f'./dataset/{channel_name}/{videoTitle}/'
            processedAudioBranchName= "audio-dataset/"
            
            getAudio(videoUrl, wholeAudioPath, wholeFileName)
            getXMLSubtitle(videoUrl, wholeSubtitlePath, wholeFileName)
            isSkipped = xMLToWordByWordSRT(wholeSubtitlePath, wholeFileName)
            pysrtModifications(wholeSubtitlePath, wholeFileName)
            splitAudioWordByWord(wholeAudioPath, wholeSubtitlePath, wholeFileName, processedAudioPath, processedAudioBranchName)
            print(f"Finished processing video: {videoTitle}")
            print(f'number of videos left are: {number_of_videos-i-1}')
            if isSkipped:
                with open(channelSkippedVideos+'.csv', 'w') as skipped_files:
                    skipped_files.write(processedAudioPath)
        print(f"Finished processing channel: {channel_url}")

    # modelFineTuning()

if __name__ == "__main__":
    main()