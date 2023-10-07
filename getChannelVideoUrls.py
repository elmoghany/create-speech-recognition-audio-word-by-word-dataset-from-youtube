from contextlib import nullcontext
from httpcore import TimeoutException
from pytube import Channel
from pytube import Playlist
from pytube import YouTube
import re
import json
import pprint
import requests
from bs4 import BeautifulSoup


# # to get the name after the @
# channel_name = urlBeforeShaping.split("@")[-1]
# print(channel_name)
# # to construct the link url
# channel_url = "https://www.youtube.com/" + "c/" + channel_name
# print(channel_url)

def getChannelVideoUrls(channel_url):
################---MODIFYING THE URL---##################
# channelNames = [    'https://www.youtube.com/@ApurvMehradr'
#                     ,'https://www.youtube.com/@Tsaog'
#                     ,'https://www.youtube.com/@MidwestOrthoatRush'
#                     ,'https://www.youtube.com/@OrthoEvalPal'
#                     ,'https://www.youtube.com/@ucsforthopaedicsurgery5365'
#                     ,'https://www.youtube.com/@DrUditKapoorortho'
#                     ,'https://www.youtube.com/@3DPhysicalTherapy'
#                     ,'https://www.youtube.com/@NewYorkOrtho'
#                     ,'https://www.youtube.com/@SummitOrthopedics1'
#                     ,'https://www.youtube.com/@RothmanOrtho'
#                     ,'https://www.youtube.com/@flortho'
#                     ,'https://www.youtube.com/@orthopaedicacademy'
#                     ,'https://www.youtube.com/@benhascientificorthotubech5611'
#                     ,'https://www.youtube.com/@universityorthopedics6220'
#                     ,'https://www.youtube.com/@TwinCitiesOrtho'
#                     ,'https://www.youtube.com/@ucteachortho7996'
#                     ,'https://www.youtube.com/@AOTraumaNorthAmerica'
#                     ,'https://www.youtube.com/@orthopaedicsurgicalvideos9653'
#                     ,'https://www.youtube.com/@cairouniversityorthopaedic4166'
#                     ,'https://www.youtube.com/@drvarunagarwalorthopaedics5089'
#                     ,'https://www.youtube.com/@panoramaorthopedics'
#                     ,'https://www.youtube.com/@seaviewortho'
#                     ,'https://www.youtube.com/@DrVinayKumarSingh'
#                     ,'https://www.youtube.com/@OrthoSurgWUSTL'
#                     ,'https://www.youtube.com/@orthooneorthopaedicspecial831'
#                     ,'https://www.youtube.com/@thecenteroregon'
#                     ,'https://www.youtube.com/@StCloudOrthopedics'
#                     ,'https://www.youtube.com/@thepaleyinstitute'
#                     ,'https://www.youtube.com/@arlingtonortho4246'
#                     ,'https://www.youtube.com/@bombayorth'
#                     ,'https://www.youtube.com/@resurgensvideo'
#                     ,'https://www.youtube.com/@kayalortho'
#                     ,'https://www.youtube.com/@WarnerOrthopedicsWellness'
#                     ,'https://www.youtube.com/@DrManujWadhwaEliteOrthopaedics'
#                     ,'https://www.youtube.com/@OrthoImplantsForLife'
#                     ,'https://www.youtube.com/@orthopedicandbalancetherap3169'
#                     ,'https://www.youtube.com/@Orthodux1'
#                     ,'https://www.youtube.com/@naileditortho2160'
#                     ,'https://www.youtube.com/@TRIAortho'
#                     ,'https://www.youtube.com/@bofas_uk'
#                     ,'https://www.youtube.com/@orthonotesdrmassoudmd.4465'
#                     ,'https://www.youtube.com/@orthopaedicneurosurgeryspe1080'
#                     ,'https://www.youtube.com/@ColumbiaOrthopedics'
#                     ,'https://www.youtube.com/@harvardglobalorthopaedicsc7762'
#                     ,'https://www.youtube.com/@SynergyOrthopedicSpecialists'
#                     ,'https://www.youtube.com/@edmondcleeman'
#                     ,'https://www.youtube.com/@orthopaedics360'
#                     ,'https://www.youtube.com/@cambridgeorthopaedics1022'
#                     ,'https://www.youtube.com/@TexasOrthopedicSpecialists'
#                     ,'https://www.youtube.com/@Pediatricorthopedic'
#                     ,'https://www.youtube.com/@orthopaedictraumasociety2877'
#                     ,'https://www.youtube.com/@ORTHOCAREDrPrashantkumar'
#                     ,'https://www.youtube.com/@drgirishguptaorthopaedicsu1755'
#                     ,'https://www.youtube.com/@miachortho'
#                     ,'https://www.youtube.com/@centerforspineandorthopedi2771'
#                     ,'https://www.youtube.com/@rapidrevisionoforthopaedics'
#                     ,'https://www.youtube.com/@carilionclinicorthopaedice3481'
#                     ,'https://www.youtube.com/@goldenstateortho918'
#                     ,'https://www.youtube.com/@DAHSAcademy'
#                     ,'https://www.youtube.com/@uvaorthopaedicsurgery2919'
#                     ,'https://www.youtube.com/@Orthopedicreview'
#                     ,'https://www.youtube.com/@totaljointorthopedics6947'
#                     ,'https://www.youtube.com/@TheYoungOrthopod'
#                     ,'https://www.youtube.com/@orthoTV'
#                     ,'https://www.youtube.com/@nabilebraheim'
#                     ,'https://www.youtube.com/@WhatsNewinOrthopedics'
#                     ,'https://www.youtube.com/@conservativeorthopedics4008'
#                     ,'https://www.youtube.com/@HuskyOrthopaedics'
#                     ,'https://www.youtube.com/@ConceptualOrthopedics'
#                     ,'https://www.youtube.com/@DrAshwaniMaichand'
#                     ,'https://www.youtube.com/@antoniowebbmd'
#                     ,'https://www.youtube.com/@OrthopaedicPrinciples'
# ]


    ################---Get ALL playlists from a CHANNEL---##################
    # Load previously processed channels and current state of playlistData
    playlistData = dict()
    playlistDS={}
    videoUrlsDS={}

    print('xxxxxxxxxxxxxxxxxxSTARTINGxxxxxxxxxxxxxxxxxxxxxxx')

    def get_playlists_from_channel(channel_url=channel_url):
        print("channel_url: ",channel_url)
        response = requests.get(channel_url + '/playlists')
        print("response: ",response)
        soup = BeautifulSoup(response.text, 'html.parser')

        playlists_ids = re.findall(r'/playlist\?list=([a-zA-Z0-9_-]+)', str(soup))

        base_url = 'https://www.youtube.com'
        playlist_urls = [f"{base_url}/playlist?list={playlist}" for playlist in playlists_ids]

        return playlist_urls

    playlists_urls = get_playlists_from_channel(channel_url)

    playlistLength = len(playlists_urls)
    print("playlist length: ", playlistLength)
    
    videoTitles=[]
    videoUrls=[] 

    playlist_number=1

    for playlist_url in playlists_urls:
        # print("         -------------------NEW PLAYLIST PROCESS-------------------------")
        # print(f'Channel     : {channel_url}')
        print(f'playlist    #  {playlist_number}/{playlistLength}')
        print("playlist url: ",playlist_url)
        playlist_number+=1

        playlist_object = Playlist(playlist_url)

        video_number=1 # counter for videos
        for video_url in playlist_object.videos:
            try:
                # print(f'video   # {video_number}/{playlist_object.length}, video url: {video_url.watch_url}')
                video_number+=1
            except ValueError as e:
                print(e)
                # print(f'video   # {video_number}/1, video url: {video_url.watch_url}')
            try:
                # print("complete title: ",video_url.title)
                videoTitle = video_url.title[:64]
                # print(" 64 char title: ",videoTitle)
                videoTitle = re.sub(r'[^\x00-\x7F]+','_', videoTitle)
                # print("language title: ",videoTitle)
                invalid_chars = '\\/:*?".<>|'
                trans = str.maketrans({char: '-' for char in invalid_chars})
                videoTitle = videoTitle.translate(trans)
                videoTitle = videoTitle.strip()
                # print("video title: ",videoTitle)
            except ValueError as e:
                videoTitle = "any video title"
            
            videoUrls.append(video_url.watch_url)
            videoTitles.append(videoTitle)
    return(videoTitles,videoUrls)