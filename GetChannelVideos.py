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

################---MODIFYING THE URL---##################
channelNames = [    'https://www.youtube.com/@ApurvMehradr'
                    ,'https://www.youtube.com/@Tsaog'
                    ,'https://www.youtube.com/@orthopaedicacademy'
                    ,'https://www.youtube.com/@MidwestOrthoatRush'
                    ,'https://www.youtube.com/@OrthoEvalPal'
                    ,'https://www.youtube.com/@ucsforthopaedicsurgery5365'
                    ,'https://www.youtube.com/@DrUditKapoorortho'
                    ,'https://www.youtube.com/@3DPhysicalTherapy'
                    ,'https://www.youtube.com/@NewYorkOrtho'
                    ,'https://www.youtube.com/@SummitOrthopedics1'
                    ,'https://www.youtube.com/@RothmanOrtho'
                    ,'https://www.youtube.com/@flortho'
                    ,'https://www.youtube.com/@universityorthopedics6220'
                    ,'https://www.youtube.com/@TwinCitiesOrtho'
                    ,'https://www.youtube.com/@ucteachortho7996'
                    ,'https://www.youtube.com/@AOTraumaNorthAmerica'
                    ,'https://www.youtube.com/@orthopaedicsurgicalvideos9653'
                    ,'https://www.youtube.com/@cairouniversityorthopaedic4166'
                    ,'https://www.youtube.com/@drvarunagarwalorthopaedics5089'
                    ,'https://www.youtube.com/@panoramaorthopedics'
                    ,'https://www.youtube.com/@seaviewortho'
                    ,'https://www.youtube.com/@DrVinayKumarSingh'
                    ,'https://www.youtube.com/@OrthoSurgWUSTL'
                    ,'https://www.youtube.com/@orthooneorthopaedicspecial831'
                    ,'https://www.youtube.com/@thecenteroregon'
                    ,'https://www.youtube.com/@StCloudOrthopedics'
                    ,'https://www.youtube.com/@thepaleyinstitute'
                    ,'https://www.youtube.com/@arlingtonortho4246'
                    ,'https://www.youtube.com/@bombayorth'
                    ,'https://www.youtube.com/@resurgensvideo'
                    ,'https://www.youtube.com/@kayalortho'
                    ,'https://www.youtube.com/@WarnerOrthopedicsWellness'
                    ,'https://www.youtube.com/@DrManujWadhwaEliteOrthopaedics'
                    ,'https://www.youtube.com/@OrthoImplantsForLife'
                    ,'https://www.youtube.com/@orthopedicandbalancetherap3169'
                    ,'https://www.youtube.com/@Orthodux1'
                    ,'https://www.youtube.com/@naileditortho2160'
                    ,'https://www.youtube.com/@TRIAortho'
                    ,'https://www.youtube.com/@bofas_uk'
                    ,'https://www.youtube.com/@orthonotesdrmassoudmd.4465'
                    ,'https://www.youtube.com/@orthopaedicneurosurgeryspe1080'
                    ,'https://www.youtube.com/@ColumbiaOrthopedics'
                    ,'https://www.youtube.com/@harvardglobalorthopaedicsc7762'
                    ,'https://www.youtube.com/@SynergyOrthopedicSpecialists'
                    ,'https://www.youtube.com/@edmondcleeman'
                    ,'https://www.youtube.com/@orthopaedics360'
                    ,'https://www.youtube.com/@cambridgeorthopaedics1022'
                    ,'https://www.youtube.com/@TexasOrthopedicSpecialists'
                    ,'https://www.youtube.com/@Pediatricorthopedic'
                    ,'https://www.youtube.com/@orthopaedictraumasociety2877'
                    ,'https://www.youtube.com/@ORTHOCAREDrPrashantkumar'
                    ,'https://www.youtube.com/@drgirishguptaorthopaedicsu1755'
                    ,'https://www.youtube.com/@miachortho'
                    ,'https://www.youtube.com/@centerforspineandorthopedi2771'
                    ,'https://www.youtube.com/@rapidrevisionoforthopaedics'
                    ,'https://www.youtube.com/@carilionclinicorthopaedice3481'
                    ,'https://www.youtube.com/@goldenstateortho918'
                    ,'https://www.youtube.com/@DAHSAcademy'
                    ,'https://www.youtube.com/@uvaorthopaedicsurgery2919'
                    ,'https://www.youtube.com/@Orthopedicreview'
                    ,'https://www.youtube.com/@totaljointorthopedics6947'
                    ,'https://www.youtube.com/@TheYoungOrthopod'
                    ,'https://www.youtube.com/@orthoTV'
                    ,'https://www.youtube.com/@nabilebraheim'
                    ,'https://www.youtube.com/@WhatsNewinOrthopedics'
                    ,'https://www.youtube.com/@conservativeorthopedics4008'
                    ,'https://www.youtube.com/@HuskyOrthopaedics'
                    ,'https://www.youtube.com/@ConceptualOrthopedics'
                    ,'https://www.youtube.com/@DrAshwaniMaichand'
                    ,'https://www.youtube.com/@antoniowebbmd'
                    ,'https://www.youtube.com/@OrthopaedicPrinciples'
]

# Checkpoint functions
def load_checkpoint():
    try:
        with open('checkpoint.json', 'r') as f:
            data = json.load(f)
            return data.get("last_channel", ""), data.get("playlistData", {})
    except FileNotFoundError:
        return "", {}

def update_checkpoint(channel_url, playlist_data):
    with open('checkpoint.json', 'w') as f:
        json.dump({"last_channel": channel_url, "playlistData": playlist_data}, f)

################---Get ALL playlists from a CHANNEL---##################
# Load previously processed channels and current state of playlistData
last_processed_channel, playlistData = load_checkpoint()
start_processing = True if last_processed_channel == "" else False

playlistData = dict()
playlistDS={}
videoUrlsDS={}

channelNumber=0
for channel_url in channelNames:
    print('xxxxxxxxxxxxxxxxxxSTARTINGxxxxxxxxxxxxxxxxxxxxxxx')
    if not start_processing:
        if channel_url == last_processed_channel:
            start_processing = True
        continue

    def get_playlists_from_channel(channel_url=channel_url):
        print("channel_url: ",channel_url)
        response = requests.get(channel_url + '/playlists')
        print("response: ",response)
        soup = BeautifulSoup(response.text, 'html.parser')
        # with open('channel_content.html', 'w', encoding='utf-8') as f:
        #     f.write(str(soup.prettify))

        playlists_ids = re.findall(r'/playlist\?list=([a-zA-Z0-9_-]+)', str(soup))

        base_url = 'https://www.youtube.com'
        playlist_urls = [f"{base_url}/playlist?list={playlist}" for playlist in playlists_ids]

        return playlist_urls

    playlists_urls = get_playlists_from_channel(channel_url)

    playlistLength = len(playlists_urls)
    print("playlist length: ", playlistLength)
    

    playlist_number=1
    for playlist_url in playlists_urls:
        # print("         -------------------NEW PLAYLIST PROCESS-------------------------")
        print(f'playlist # {playlist_number}/{playlistLength}, Channel: {channel_url}')
        print("playlist url: ",playlist_url)
        playlist_number+=1

        totalPlaylistLength = 0
        playlist_object = Playlist(playlist_url)

        video_number=1 # counter for videos
        videoUrls=[] 
        videoTitles=[]
        videoLength=[]
        videoViews=[]
        for video_url in playlist_object.videos:
            #logging to screen only
            try:
                print(f'video # {video_number}/{playlist_object.length}, channel: {video_url.watch_url}')
                video_number+=1
            except ValueError as e:
                print(e)
                print(f'video # {video_number}/1, channel: {video_url.watch_url}')
            # print(video_url.title)
            try:
                videoTitle = video_url.title
            except ValueError as e:
                videoTitle = "any video title"

            videoUrlsDS[video_url.watch_url]= { 
                    "videoTitles" : videoTitle,
                    "videoLength" : video_url.length / 60 / 60,
                    "videoViews"  : video_url.views
            }
            totalPlaylistLength += video_url.length / 60 / 60
        try:
            playlistViews = playlist_object.views
        except ValueError as e:
            playlistViews = 1
        else:
            playlistViews = 1
        try:
            playlistVideosCount = playlist_object.length
        except ValueError as e:
            playlistVideosCount = 1
        try:
            playlistTitle = playlist_object.title
        except ValueError as e:
            playlistTitle = "any playlist title"
        
        playlistDS[playlist_url] = {  
                                    "playlistViewsCount"    : playlistViews, 
                                    "playlistTitle"         : playlistTitle, 
                                    "playlistVideosCount"   : playlistVideosCount,
                                    "playlistTotalHours"    : totalPlaylistLength,
                                    "videoUrl"              : videoUrlsDS
        }
        
    playlistData[channel_url] = playlistDS

    # Update checkpoint after successfully processing the channel
    channelNumber+=1
    if(channelNumber%5==0):
        update_checkpoint(channel_url, playlistData)

print("         -------------------SAVING PLAYLIST-------------------------")
save_file=open('channelsDetails.json','a')
json.dump(playlistData, save_file, indent=4)
save_file.close()
print("         ###################SAVED PLAYLIST#########################")
print("xxxxxxxxxxxxxxxxxxxxENDINGxxxxxxxxxxxxxxxxxxxxxxxxxxxx")





# for channel_url in channelNames:
#     print('xxxxxxxxxxxxxxxxxxSTARTINGxxxxxxxxxxxxxxxxxxxxxxx')

#     def get_playlists_from_channel(channel_url=channel_url):
#         print("channel_url: ",channel_url)
#         response = requests.get(channel_url + '/playlists')
#         print("response: ",response)
#         soup = BeautifulSoup(response.text, 'html.parser')
#         # with open('channel_content.html', 'w', encoding='utf-8') as f:
#         #     f.write(str(soup.prettify))

#         playlists_ids = re.findall(r'/playlist\?list=([a-zA-Z0-9_-]+)', str(soup))

#         base_url = 'https://www.youtube.com'
#         playlist_urls = [f"{base_url}/playlist?list={playlist}" for playlist in playlists_ids]

#         return playlist_urls

#     playlists_urls = get_playlists_from_channel(channel_url)

#     playlistLength = len(playlists_urls)
#     print("playlist length: ", playlistLength)
    
#     playlistData = dict()
#     playlistData = {}

#     playlist_number=1
#     for playlist_url in playlists_urls:
#         print("         -------------------NEW PLAYLIST PROCESS-------------------------")
#         print(f'playlist # {playlist_number}/{playlistLength}, Channel: {channel_url}')
#         print("playlist url: ",playlist_url)
#         playlist_number+=1

#         totalPlaylistLength = 0
#         playlist_object = Playlist(playlist_url)
#         playlistData.setdefault(playlist_url,{})['playlistTitle'] = playlist_object.title
#         playlistData[playlist_url]['playlistViewsCount'] = playlist_object.views
#         video_number=1 # counter for videos
#         for video_url in playlist_object.videos:
#             try:
#                 print(f'video # {video_number}/{playlist_object.length}, channel: {channel_url}')
#                 video_number+=1
#             except ValueError as e:
#                 print(e)
#                 print(f'video # {video_number}/1, channel: {channel_url}')

#             #creating video_url object
#             nested_video = playlistData.setdefault(playlist_url,{})
#             nested_video.setdefault(video_url.watch_url,{})['videoTitle'] = video_url.title
#             playlistData.setdefault(playlist_url,{})[video_url.watch_url]['videoViewsCount'] = video_url.views
#             playlistData.setdefault(playlist_url,{})[video_url.watch_url]['videoLength'] = video_url.length / 60 / 60
#             totalPlaylistLength += video_url.length / 60 / 60
#         playlistData[playlist_url]['playlistVideosCount'] = len(playlist_object.videos)
#         playlistData[playlist_url]['playlistTotalHours'] = totalPlaylistLength
#         print("         -------------------SAVING PLAYLIST-------------------------")
#         save_file=open('channelsDetails.json','a')
#         json.dump(playlistData, save_file, indent=4)
#         save_file.close()
#         print("         ###################SAVED PLAYLIST#########################")
#     print("xxxxxxxxxxxxxxxxxxxxENDINGxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

################---Get 1 video title from 1 playlist---#################
# playlistData = {
#     '<playlistUrls>':  [{
#         'playlistTitle': None,
#         'playlistViewsCount': None,
#         'playlistVideosCount': None,
#         '<videoUrls>': [{
#             'videoTitle': None,
#             'videoViewsCount': None,
#             'videoLength': None,
#         }]
#     }]
# }


