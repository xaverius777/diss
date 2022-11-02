from pydub import AudioSegment
from pydub.playback import play
import os
input_dir="C:/Users/javic/Downloads/mixing_data/listen_for_testing/"
output_dir="C:/Users/javic/Downloads/rome/"
# file_list = [input_dir + file for file in os.listdir(input_dir)]
file_list = [file for file in os.listdir(input_dir)]
file_list.remove("scp21817")
for file in file_list:
    print(f"Processing audio {file}")
    audio_in_file = input_dir + file
    # audio_out_file = output_dir + file.replace(".wav", "_silence_padded.wav")
    audio_out_file = output_dir + file
    # half_a_sec = AudioSegment.silent(duration=500)
    # one_hundred_mil = AudioSegment.silent(duration=100)
    # one_mil = AudioSegment.silent(duration=1)
    audio = AudioSegment.from_wav(audio_in_file)
    half_remainder = (10 - audio.duration_seconds)/2
    half_remainder = AudioSegment.silent(duration=half_remainder * 1001)
    # print(half_remainder)
    audio = half_remainder + audio + half_remainder
    audio = audio[:10000]
    if audio.duration_seconds != 10.0:
        raise Exception("Audio length is not ten seconds!")
    audio.export(audio_out_file, format="wav")
print("Done, all files were padded with silence at both edges")