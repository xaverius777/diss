import os
from scipy import signal
import numpy as np
import soundfile as sf
import re
# Get noisy speech files, divide them in subgroups
noisy_dir = "C:/Users/javic/Downloads/mixing_data/val_noisy/"
storage_dir = "C:/Users/javic/Downloads/rome/"
# noisy_filename_list = [noisy_dir + file for file in os.listdir(noisy_dir)]
noisy_filename_list = [file for file in os.listdir(noisy_dir)]
# testing_noisy_file = "C:/Users/javic/Downloads/mixing_data/noisy_speech/female_american_6_construction_noise_2_snr_40_rms_-33.wav"
# initialize = np.zeros(0)
# noisy_audio, noisy_audio_fs = sf.read(testing_noisy_file, start=0, stop=None)
first_group = re.compile(r".*(((fema)|(ma))le)_[a-z]+_[1-4]_.*")
second_group = re.compile(r".*(((fema)|(ma))le)_[a-z]+_[5-7]_.*")
third_group = re.compile(r".*(((fema)|(ma))le)_[a-z]+_([8-9]|10)_.*")
first_group_list = list(filter(first_group.match, noisy_filename_list))
second_group_list = list(filter(second_group.match, noisy_filename_list))
third_group_list = list(filter(third_group.match, noisy_filename_list))
# print(third_group_list)
# Get the impulse responses
rir_dir = "C:/Users/javic/Downloads/air_database_release_1_4/AIR_1_4/AIR_wav_files_downsampled/"
rir_filename_list = [rir_dir + file for file in os.listdir(rir_dir)]
booth_rir_file = "C:/Users/javic/Downloads/air_database_release_1_4/AIR_1_4/AIR_wav_files_downsampled/air_booth_0_1_3.wav"
meeting_rir_file = "C:/Users/javic/Downloads/air_database_release_1_4/AIR_1_4/AIR_wav_files_downsampled/air_meeting_0_1_3.wav"
office_rir_file = "C:/Users/javic/Downloads/air_database_release_1_4/AIR_1_4/AIR_wav_files_downsampled/air_office_0_1_3.wav"
lecture_rir_file = "C:/Users/javic/Downloads/air_database_release_1_4/AIR_1_4/AIR_wav_files_downsampled/air_lecture_0_1_3.wav"
booth_rir_audio, booth_rir_audio_fs = sf.read(booth_rir_file, start=0, stop=None)
meeting_rir_audio, meeting_rir_audio_fs = sf.read(meeting_rir_file, start=0, stop=None)
office_rir_audio, office_rir_audio_fs = sf.read(office_rir_file, start=0, stop=None)
lecture_rir_audio, lecture_rir_audio_fs = sf.read(lecture_rir_file, start=0, stop=None)
# print(booth_rir_file, booth_rir_audio.shape, booth_rir_audio)
# Loop over the groups and convolve the files with the appropiate impulse response.
# Group 1 = meeting; group 2 = office ; group 3 = lecture
for file in first_group_list:
    input_path = noisy_dir + file
    output_path = storage_dir + file.replace(".wav", "_r_meeting.wav")
    print(f"File {os.path.basename(input_path)} will be reverberated \n It will be written to {os.path.basename(output_path)}")
    noisy_audio, noisy_audio_fs = sf.read(input_path, start=0, stop=None)
    reverbed_file = signal.convolve(noisy_audio, meeting_rir_audio)
    sf.write(output_path, reverbed_file, samplerate=16000)
for file in second_group_list:
    input_path = noisy_dir + file
    output_path = storage_dir + file.replace(".wav", "_r_office.wav")
    print(f"File {os.path.basename(input_path)} will be reverberated \n It will be written to {os.path.basename(output_path)}")
    noisy_audio, noisy_audio_fs = sf.read(input_path, start=0, stop=None)
    reverbed_file = signal.convolve(noisy_audio, office_rir_audio)
    sf.write(output_path, reverbed_file, samplerate=16000)
for file in third_group_list:
    input_path = noisy_dir + file
    output_path = storage_dir + file.replace(".wav", "_r_lecture.wav")
    print(f"File {os.path.basename(input_path)} will be reverberated \n It will be written to {os.path.basename(output_path)}")
    noisy_audio, noisy_audio_fs = sf.read(input_path, start=0, stop=None)
    reverbed_file = signal.convolve(noisy_audio, lecture_rir_audio)
    sf.write(output_path, reverbed_file, samplerate=16000)
