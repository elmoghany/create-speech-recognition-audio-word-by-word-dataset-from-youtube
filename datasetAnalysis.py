import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Initialize an empty DataFrame to store all metadata
all_metadata = pd.DataFrame(columns=["path", "transcription"])

# Iterate over the dataset directories
for channel in os.listdir('./dataset/'):
    for video in os.listdir(f'./dataset/{channel}'):
        if not video.endswith(".csv"):
            current_path = f'./dataset/{channel}/{video}/'
            current_metadata_path = os.path.join(current_path, 'metadata.csv')
            if os.path.exists(current_metadata_path):
                # Read the CSV and append it to the main dataframe
                metadata = pd.read_csv(current_metadata_path, encoding="ISO-8859-1")
                all_metadata = pd.concat([all_metadata, metadata], ignore_index=True)

# Calculate the frequency of each transcription
transcription_counts = all_metadata['transcription'].value_counts()

# Get the top 50
# top_50_transcriptions = transcription_counts.head(50)
top_50_transcriptions = transcription_counts.iloc[351:400]

# Plotting
plt.figure(figsize=(15, 10))
top_50_transcriptions.plot(kind='barh', color='skyblue')
plt.gca().invert_yaxis()  # to display the word with the highest frequency at the top
plt.xlabel('Frequency')
plt.ylabel('Transcription Word')
plt.title('Top 50 Transcription Word Frequencies')
plt.show()


# for channel in os.listdir('./dataset/'):
#     print("channel: ", channel)
#     for video in os.listdir(f'./dataset/{channel}'):
#         if not video.endswith(".csv"):
#             print("video: ", video)
#             if os.path.isdir(f'./dataset/{channel}/{video}/audio-dataset'):
#                 current_path = f'/dataset/{channel}/{video}/'
#                 current_metadata = os.path.join(current_path, 'metadata.csv')
#                 # print("this video has audio dataset: ", video)
#                 print(f'current_path: {current_path}')
#                 print(f'current_metadata: {current_metadata}')

                
#     skip_path = os.path.join(subtitle_file_path, 'skip.xml')
#     if (os.path.isfile(skip_path)):
#         print("No splits are done - skip")

# # Read existing JSON file
# with open('new_data.json', 'r') as json_file:
#     new_data_dicts = json.load(json_file)
    
#     # Loop through each entry in the list of dictionaries
#     for data_dict in new_data_dicts:
#         print("sampling & creating dataset row:")
#         path = data_dict["path"]
#         transcription = data_dict["transcription"]
        
#         # Here, audio information is already available in data_dict["audio"]
#         new_sampling_rate = data_dict["audio"]["sampling_rate"]
#         audio_data = np.array(data_dict["audio"]["array"])
# print(new_data_dicts[:1])
