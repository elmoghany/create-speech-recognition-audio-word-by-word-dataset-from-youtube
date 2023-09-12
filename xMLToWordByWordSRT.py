#xml_to_word_by_word.py
import os
import xml.etree.ElementTree as ET
import re

def xMLToWordByWordSRT(filename, subtitle_file_path):
    def convert_xml_to_word_by_word_srt(xml_content):
        root = ET.fromstring(xml_content)
        srt_lines = []
        line_num = 1

        for child in root:
            if child.tag == 'body':
                p_elements = list(child.iter('p'))  # Find all <p> elements within the body
                for i, p in enumerate(p_elements):
                    words = [s for s in p if s.tag == 's']
                    p_start_time = int(p.get('t'))  # Timing for the current <p> tag

                    # Get the 'd' attribute value of the next <p> element if available
                    next_p_duration = int(p_elements[i + 1].get('t')) - int(p.get('t')) if i < len(p_elements) - 1 else 0

                    for i, word in enumerate(words):
                        if 't' in word.attrib:
                            word_start_time = int(word.get('t')) + p_start_time  # Use the provided timing for each word
                        else:
                            # Calculate timing for words without 't' attribute based on the <p>
                            word_start_time = p_start_time
                        word_text = word.text.strip()
                        
                        # Calculate the end time of the current word
                        if i < len(words) - 1:
                            next_word = words[i + 1]
                            if 't' in next_word.attrib:
                                word_end_time = int(next_word.get('t')) + p_start_time
                            else:
                                word_end_time = p_start_time
                        else:
                            # Last word in the <p>
                            word_end_time = p_start_time + next_p_duration

                        srt_lines.append(f"{line_num}\n{format_time(word_start_time)} --> {format_time(word_end_time)}\n{word_text}\n")
                        line_num += 1

        return '\n'.join(srt_lines)

    def format_time(milliseconds):
        seconds, milliseconds = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"

    download_path = os.path.join(subtitle_file_path, filename)
    if not os.path.exists(subtitle_file_path):
        os.makedirs(subtitle_file_path)

    print('download subtitle path: ', download_path)

    # Your XML string
    with open(download_path+'.xml', 'r') as xml_file:
        print('********inside XML To Word By Word SRT********')
        xml_content=xml_file.read()

    # Convert to word-by-word SRT and save to file
    word_by_word_srt = convert_xml_to_word_by_word_srt(xml_content)
    with open(download_path+'_word_by_word.srt', 'w') as srt_file:
        srt_file.write(word_by_word_srt)

    print("XML captions converted to SRT format and saved to 'captions.srt'")
