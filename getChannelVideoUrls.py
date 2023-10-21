from pytube import Playlist
import re
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
# channelNames = [    

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