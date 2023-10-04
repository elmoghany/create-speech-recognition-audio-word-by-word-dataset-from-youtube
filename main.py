#main.py
from getAudio import getAudio
from getXMLSubtitle import getXMLSubtitle 
from xMLToWordByWordSRT import xMLToWordByWordSRT
from splitAudioWordByWord import splitAudioWordByWord
from pysrtModifications import pysrtModifications
from modelFineTuning import modelFineTuning
from GetChannelVideos import GetChannelVideos

def main():
    youtube_url='https://www.youtube.com/watch?v=ofiTvFWkuLM'
    filename='xyz'
    channelUrl='https://www.youtube.com/@orthoTV'
    GetChannelVideos()
    #getAudio(youtube_url, filename, './audio/')
    #getXMLSubtitle(youtube_url, filename, './subtitle/')
    #xMLToWordByWordSRT(filename, './subtitle/')
    #splitSRTIntoMultipleFiles(filename, './subtitle/')
    #splitAudioWordByWord(filename, './audio/', './subtitle/', './dataset/')
    #pysrtModifications(filename, './subtitle/')
    # modelFineTuning()

if __name__ == "__main__":
    main()