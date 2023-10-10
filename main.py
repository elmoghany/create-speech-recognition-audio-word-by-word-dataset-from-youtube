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
import multiprocessing
import time 
# Done
                    # 'https://www.youtube.com/@centerforspineandorthopedi2771'
                    # ,'https://www.youtube.com/@miachortho'
                    # ,'https://www.youtube.com/@drgirishguptaorthopaedicsu1755'
                    # 'https://www.youtube.com/@Tsaog'
                    # 'https://www.youtube.com/@goldenstateortho918'
# 250 replace pop                    'https://www.youtube.com/@orthopaedicacademy'

# def main():

def process_channel(channel_url,queue):
    start_time = time.time()
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
        videoTitle = videoTitles.pop() #need to get it from the first inputs!!
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
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Put the elapsed time in the queue
    queue.put((channel_url, time.ctime(start_time), elapsed_time))

    # modelFineTuning()

if __name__ == "__main__":

    channelNames=[  
                    # 'https://www.youtube.com/@ORTHOCAREDrPrashantkumar'
                    'https://www.youtube.com/@carilionclinicorthopaedice3481'
                    ,'https://www.youtube.com/@DAHSAcademy'
                    ,'https://www.youtube.com/@uvaorthopaedicsurgery2919'
                    ,'https://www.youtube.com/@Orthopedicreview'
                    ,'https://www.youtube.com/@totaljointorthopedics6947'
                    ,'https://www.youtube.com/@TheYoungOrthopod'
                ]
    
    # Create a multiprocessing queue to store elapsed times
    queue = multiprocessing.Queue()

    processes = []

    for channel in channelNames:
        process = multiprocessing.Process(target=process_channel, args=(channel,queue))
        process.start()
        processes.append(process)
        print(f"+++++++++++++++++process {process} started+++++++++++++++++++++")

    for process in processes:
        process.join()
        print(f"+++++++++++++++++process {process} ended++++++++++++++++++++++")

    # Retrieve and print elapsed times for each process
    while not queue.empty():
        channel_url, start_time, elapsed_time = queue.get()
        print(f"Time taken for {channel_url}: {elapsed_time:.2f//60//60} seconds and it started: {start_time:.2f}")
