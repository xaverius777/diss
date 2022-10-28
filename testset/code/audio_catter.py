import os
import random
import argparse
from moviepy.editor import concatenate_audioclips, AudioFileClip
def concatenate_audio_moviepy(input_dir, output_dir, output_name, duration, no_files, extension, loop):
    """Concatenates several audio files into one audio file using MoviePy
    and save it to `output_dir`. Must specify extension (e.g. .mp3, .wav)"""
    file_list = os.listdir(input_dir)
    file_list = [input_dir + file for file in file_list]
    if loop == False:
        print("The files will be concatenated in random order as many times as the no_files argument specifies")
        for number in range(no_files):
            output_path = output_dir + output_name + f"_{number + 1}{extension}"
            random.shuffle(file_list)
            clip_list= []
            for clip in file_list:
                clip_list.append(AudioFileClip(clip))
                final_clip = concatenate_audioclips(clip_list)
                print(f"Audio currently is {final_clip.duration} seconds long")
                if final_clip.duration > duration:
                    print("Reached max duration, let's write this to a file")
                    break
            final_clip.write_audiofile(output_path)
    else:
        for clip in file_list:
            print("The files will be looped")
            output_path = output_dir + output_name + f"_{file_list.index(clip) + 1}{extension}"
            clip_list=[]
            while True:
                clip_list.append(AudioFileClip(clip))
                final_clip = concatenate_audioclips(clip_list)
                print(f"Audio currently is {final_clip.duration} seconds long")
                if final_clip.duration > duration:
                    print("Reached max duration, let's write this to a file")
                    break
            final_clip.write_audiofile(output_path)
parser = argparse.ArgumentParser(description = "make audio selection process easier")
parser.add_argument("-i", "--input_dir", help="The directory where the files you want to concatenate are located", type=str)
parser.add_argument("-o", "--output_dir", help="The directory where you want to store your concatenated files", type=str)
parser.add_argument("-d", "--duration", help="When the duration of the concatenated audios surpass this number, they"\
                    "will be written to a file", type=int)
parser.add_argument("-n", "--no_files", help="The number of concatenated files you want", type=int)
parser.add_argument("-e", "--extension", help="The extension the concatenated audio files will have. E.g., .mp3, .wav", type=str)
parser.add_argument("-m", "--output_name", help = "The name you want your concatenated files to have. It will be followed by their number."\
                    "For example, if you make 3 files with the name train, their filenames will be train_1, train_2, train_3", type=str)
parser.add_argument("-l", "--loop", help = "Instead of concatenating the audios in the input directory, loop each one of them"\
                    "for the specified duration", action='store_true')
args = parser.parse_args()
if __name__ == '__main__':
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_name = args.output_name
    duration = args.duration
    no_files =  args.no_files
    extension = args.extension
    loop = args.loop
    print("WARNING: this program only concatenates, it doesn't do fade in or fade off. For it to work,"\
          "the files must begin and end in silence.")
    concatenate_audio_moviepy(extension = extension, no_files = no_files, loop= loop, \
    input_dir = input_dir, output_dir = output_dir, duration = duration, output_name= output_name,)
