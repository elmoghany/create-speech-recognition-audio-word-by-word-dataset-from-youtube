#getSubtitle.py
import pytube
import os

def getXMLSubtitle(youtube_url, filename, subtitle_file_path):
    #download & save captions
    print('********inside getting xml subtitle********')
    yt = pytube.YouTube(youtube_url)
    
    print('applying caption language: ')
    #bypass_age_gate() is a fix for yt.captions['a.en']
    len(yt.streams)
    yt.bypass_age_gate()
    caption = yt.captions['a.en']

    print('generate xml captions')
    caption_xml = caption.xml_captions
    
    print('generate srt captions')
    caption_srt = caption.generate_srt_captions()
    
    download_path = os.path.join(subtitle_file_path, filename)
    if not os.path.exists(subtitle_file_path):
        os.makedirs(subtitle_file_path)

    print('download subtitle path: ', download_path)

    print('saving caption files')
    with open(download_path+".xml","w") as xml_file:
        xml_file.write(caption_xml)
    with open(download_path+".srt","w") as srt_file:
        srt_file.write(caption_srt)