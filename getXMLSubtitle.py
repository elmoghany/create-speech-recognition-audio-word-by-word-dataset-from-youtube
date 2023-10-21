#getSubtitle.py
import pytube
import os

def getXMLSubtitle(youtube_url, subtitle_file_path, filename):
    #download & save captions
    print('********2) inside getting xml subtitle********')
    yt = pytube.YouTube(youtube_url,use_oauth=True, allow_oauth_cache=True)
    
    download_path = os.path.join(subtitle_file_path, filename)
    if not os.path.exists(subtitle_file_path):
        os.makedirs(subtitle_file_path)
    # print('download subtitle path: ', download_path)

    # print('applying caption language: ')
    #bypass_age_gate() is a fix for yt.captions['a.en']
    len(yt.streams)
    yt.bypass_age_gate()

    try:
        caption = yt.captions['a.en']

        # print('generate xml captions')
        caption_xml = caption.xml_captions
        
        # print('generate srt captions')
        # caption_srt = caption.generate_srt_captions()

        with open(download_path+".xml","w") as xml_file:
            xml_file.write(caption_xml)
        # with open(download_path+".srt","w") as srt_file:
        #     srt_file.write(caption_srt)
        print('captions saved successfully')

    except (KeyError) as e:
        print("KeyError: ",e)
        download_path = os.path.join(subtitle_file_path, 'skip')
        if not os.path.exists(subtitle_file_path):
            os.makedirs(subtitle_file_path)

        with open(download_path+".xml","w") as xml_file:
            xml_file.write('')

