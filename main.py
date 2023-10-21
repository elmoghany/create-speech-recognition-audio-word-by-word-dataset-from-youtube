#main.py
from getAudio import getAudio
from getXMLSubtitle import getXMLSubtitle 
from xMLToWordByWordSRT import xMLToWordByWordSRT
from splitAudioWordByWord import splitAudioWordByWord
from pysrtModifications import pysrtModifications
# from modelFineTuning import modelFineTuning
from getChannelVideoUrls import getChannelVideoUrls
import multiprocessing
import time 
# Done
                    # ,'https://www.youtube.com/@Orthopedicreview'
                    # ,'https://www.youtube.com/@ColumbiaOrthopedics'
                    # ,'https://www.youtube.com/@harvardglobalorthopaedicsc7762'
                    # ,'https://www.youtube.com/@TheYoungOrthopod'
                    # ,'https://www.youtube.com/@centerforspineandorthopedi2771'
                    # ,'https://www.youtube.com/@miachortho'
                    # ,'https://www.youtube.com/@drgirishguptaorthopaedicsu1755'
                    # ,'https://www.youtube.com/@Tsaog'
                    # ,'https://www.youtube.com/@goldenstateortho918'
                    # ,'https://www.youtube.com/@totaljointorthopedics6947'
                    # ,'https://www.youtube.com/@uvaorthopaedicsurgery2919'
                    # ,'https://www.youtube.com/@carilionclinicorthopaedice3481'
                    # ,'https://www.youtube.com/@DAHSAcademy'
                    # ,'https://www.youtube.com/@WhatsNewinOrthopedics'
                    # ,'https://www.youtube.com/@conservativeorthopedics4008'
                    # ,'https://www.youtube.com/@orthopaedicneurosurgeryspe1080'
                    # ,'https://www.youtube.com/@nabilebraheim'
                    # ,'https://www.youtube.com/@ConceptualOrthopedics'
                    # ,'https://www.youtube.com/@DrAshwaniMaichand'
                    # ,'https://www.youtube.com/@orthonotesdrmassoudmd.4465'
                    # ,'https://www.youtube.com/@antoniowebbmd'
                    # ,'https://www.youtube.com/@OrthopaedicPrinciples'
                    # ,'https://www.youtube.com/@Pediatricorthopedic'
                    # ,'https://www.youtube.com/@TexasOrthopedicSpecialists'
                    # ,'https://www.youtube.com/@rapidrevisionoforthopaedics'
                    # ,'https://www.youtube.com/@SynergyOrthopedicSpecialists'
                    # ,'https://www.youtube.com/@TRIAortho'
                    # ,'https://www.youtube.com/@orthopaedictraumasociety2877'
                    # ,'https://www.youtube.com/@orthopaedics360'
                    # ,'https://www.youtube.com/@cambridgeorthopaedics1022'
                    # ,'https://www.youtube.com/@edmondcleeman'
                    # ,'https://www.youtube.com/@MidwestOrthoatRush'
                    # ,'https://www.youtube.com/@HuskyOrthopaedics'
                    # ,'https://www.youtube.com/@ORTHOCAREDrPrashantkumar'
                    # ,'https://www.youtube.com/@DrUditKapoorortho'
                    # ,'https://www.youtube.com/@orthooneorthopaedicspecial831'
                    # ,'https://www.youtube.com/@drvarunagarwalorthopaedics5089'
                    # ,'https://www.youtube.com/@resurgensvideo'
                    # ,'https://www.youtube.com/@DrManujWadhwaEliteOrthopaedics'
                    # ,'https://www.youtube.com/@StCloudOrthopedics'
                    # ,'https://www.youtube.com/@naileditortho2160'
                    # ,'https://www.youtube.com/@Orthodux1'
                    # ,'https://www.youtube.com/@orthopedicandbalancetherap3169'
                    # ,'https://www.youtube.com/@thepaleyinstitute'
                    # ,'https://www.youtube.com/@WarnerOrthopedicsWellness'
                    # ,'https://www.youtube.com/@NewYorkOrtho'
                    # ,'https://www.youtube.com/@kayalortho'
                    # ,'https://www.youtube.com/@ucsforthopaedicsurgery5365'
                    # ,'https://www.youtube.com/@ApurvMehradr'
                    # ,'https://www.youtube.com/@3DPhysicalTherapy'
                    # ,'https://www.youtube.com/@SummitOrthopedics1'
                    # ,'https://www.youtube.com/@RothmanOrtho'
                    # ,'https://www.youtube.com/@flortho'
                    # ,'https://www.youtube.com/@benhascientificorthotubech5611'
                    # ,'https://www.youtube.com/@TwinCitiesOrtho'
                    # ,'https://www.youtube.com/@AOTraumaNorthAmerica'
                    # ,'https://www.youtube.com/@orthopaedicsurgicalvideos9653'
                    # ,'https://www.youtube.com/@cairouniversityorthopaedic4166'
                    # ,'https://www.youtube.com/@panoramaorthopedics'
                    # ,'https://www.youtube.com/@seaviewortho'
                    # ,'https://www.youtube.com/@OrthoSurgWUSTL'
                    # ,'https://www.youtube.com/@thecenteroregon'
                    # ,'https://www.youtube.com/@arlingtonortho4246'
                    # ,'https://www.youtube.com/@bombayorth'
                    # ,'https://www.youtube.com/@OrthoImplantsForLife'





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
    print(f"number of videos {len(videoUrls)} for channel {channel_name}")

    number_of_videos = (len(videoUrls))
    match channel_name:
        case 'orthoTV':
            videosLeft = 842
        case 'ucteachortho7996':
            videosLeft = 755
        case 'orthopaedicacademy':
            videosLeft = 231
        case 'universityorthopedics6220':
            videosLeft = 116
        case 'OrthoEvalPal':
            videosLeft = 67
        case _:
            videosLeft = 10000
            
    video_queue = multiprocessing.Queue()
    video_processes = []

    for i in range(len(videoUrls)):
        print("****************************PROCESSING VIDEO in CHANNEL*********************************")
        print("channel name: ",channel_name)
        if  (number_of_videos - i) > videosLeft:
            videoTitle = videoTitles.pop() #need to get it from the first inputs!!
            videoUrl = videoUrls.pop()
            print("cont.")
            continue   
        videoTitle = videoTitles.pop() #need to get it from the first inputs!!
        videoUrl = videoUrls.pop()
        wholeFileName="whole"
        channelPath=f'./dataset/{channel_name}/'
        channelSkippedVideos=f'./dataset/{channel_name}/{channel_name}'
        wholeSubtitlePath=f'./dataset/{channel_name}/{videoTitle}/wholeSubtitle/'
        wholeAudioPath=f'./dataset/{channel_name}/{videoTitle}/wholeAudio/'
        processedAudioPath=f'./dataset/{channel_name}/{videoTitle}/'
        processedAudioBranchName= "audio-dataset/"
        
        hasAudio = getAudio(videoUrl, wholeAudioPath, wholeFileName)
        if hasAudio == True:
            getXMLSubtitle(videoUrl, wholeSubtitlePath, wholeFileName)
            isSkipped = xMLToWordByWordSRT(wholeSubtitlePath, wholeFileName)
            pysrtModifications(wholeSubtitlePath, wholeFileName)
            splitAudioWordByWord(wholeAudioPath, wholeSubtitlePath, wholeFileName, processedAudioPath, processedAudioBranchName)
            print("isSkipped= ", isSkipped)
            if isSkipped:
                with open(channelSkippedVideos+'.csv', 'w') as skipped_files:
                    skipped_files.write(processedAudioPath)
        print(f"6) Finished processing video: {videoTitle}")
        print(f'6) number of videos left are: {number_of_videos-i-1} @{channel_name}')
    print(f"Finished processing channel: {channel_url}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Put the elapsed time in the queue
    queue.put((channel_url, time.ctime(start_time), elapsed_time))

    # modelFineTuning()

if __name__ == "__main__":

#orthopaedicacademy
#NewYorkOrtho
    channelNames=[  
                    'https://www.youtube.com/@orthoTV'
                    ,'https://www.youtube.com/@orthopaedicacademy'
                    ,'https://www.youtube.com/@OrthoEvalPal'
                    ,'https://www.youtube.com/@universityorthopedics6220'
                    ,'https://www.youtube.com/@ucteachortho7996'
                ]
    
    # Create a multiprocessing queue to store elapsed times
    queue = multiprocessing.Queue()

    processes = []

    for channel in channelNames:
        process = multiprocessing.Process(target=process_channel, args=(channel,queue))
        process.start()
        processes.append(process)
        print(f"++++++++process {process} for channel {channel} started+++++++++++")

    for process in processes:
        process.join()
        print(f"+++++++process {process} ended+++++++++++++")

    # Retrieve and print elapsed times for each process
    while not queue.empty():
        channel_url, start_time, elapsed_time = queue.get()
        print(f"Time taken for {channel_url}: {elapsed_time//60//60} seconds and it started: {start_time}")

    # for channel in channelNames:
        # process_channel(channel,queue)