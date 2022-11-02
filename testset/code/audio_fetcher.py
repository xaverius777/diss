print("WARNING: DOESN'T WORK IN WINDOWS BECAUSE OF LIBROSA ISSUES. FIX THAT OR USE ANOTHER OS")
import argparse
import pandas as pd
import librosa
import shutil
def main(csv_path, audio_directory, storage_directory, number_of_files, characteristics, duration):
    csv_path = csv_path
    audio_directory = audio_directory
    number_of_files = number_of_files
    characteristics = characteristics
    duration = duration
    information_string = "There are "
    information_string_dict = {
    "age_1": "in their ",
    "accent_1": "with an accent from ",
    "age_2": "are in their ",
    "accent_2": "have an accent from "
    }
    duration_start, duration_end = duration[0], duration[1]
    characteristic_counter = 1
    df = pd.read_csv(csv_path)
    characteristic_index = 0
    while characteristic_index != len(characteristics):
        characteristic_index_0 = characteristic_index
        characteristic_index_1 = characteristic_index + 1
        df = df.loc[df[f"{characteristics[characteristic_index_0]}"] == f"{characteristics[characteristic_index_1]}"]
        if characteristic_counter == 1:
            if characteristics[characteristic_index_0] == "gender":
                information_string += f"{len(df)} {characteristics[characteristic_index_1]} speakers, "
            elif characteristics[characteristic_index_0] == "age" or characteristics[characteristic_index_0] == "accent":
                information_string += f"{len(df)} speakers {information_string_dict[f'{characteristics[characteristic_index_0]}_1']}{characteristics[characteristic_index_1]}, "
            else:
                pass
        if characteristic_counter > 1:
            information_string += "out of which "
            if characteristics[characteristic_index_0] == "gender":
                information_string += f"{len(df)} are {characteristics[characteristic_index_1]}, "
            elif characteristics[characteristic_index_0] == "age" or characteristics[characteristic_index_0] == "accent":
                information_string += f"{len(df)} {information_string_dict[f'{characteristics[characteristic_index_0]}_2']}{characteristics[characteristic_index_1]}, "
        characteristic_counter += 1
        characteristic_index += 2
    information_string = information_string[:-2] + "."
    print(information_string)
    desired_audios = []
    for row in range(len(df)):
        filename = (df.iloc[row]["filename"]).split("/")[1]
        audio_path = audio_directory + filename
        #convert file to .wav

        try:
            file_duration = int(librosa.get_duration(filename=audio_path))
            print(f"Checking if audio '{audio_path}' meets the duration requirement and we can add it to the list of desired audios...")
            if file_duration in range(duration_start, duration_end, 1):
                desired_audios.append(audio_path)
                print(f"Added! Out of the desired {number_of_files} audios, we have {len(desired_audios)}")
            else:
                print("Discarded")
        except FileNotFoundError:
            print(f"Audio '{audio_path}' not found, we continue")
        if len(desired_audios) == number_of_files:
            break
    print(f"The desired files were successfully extracted!") 
    for audio_file in desired_audios:
        shutil.copy(audio_file, storage_directory)
        print(f"Audio '{audio_file}' has been put in the storage directory")
    print("Your files were successfully extracted!")
parser = argparse.ArgumentParser(description = "make audio selection process easier")
parser.add_argument("-n", "--number_of_files", help="How many files of the specified characteristics you want", type=int)
parser.add_argument("-l", "--list_of_characteristics", help="The characteristics you want your files to have." \
                          "Specify them in this format: [[column name, desired value][column name, desired value]...]" \
                          "Given a dataset, the program will make a subset of it which meets all the constraints you specified in order." \
                          "For example, if you specify [[gender, female][accent, us][age, forties]] you will get the speakers that meet those constraints in order:" \
                          "The female speakers who are American who are in their forties.",
                    type=str, nargs ='+')
parser.add_argument("-d", "--duration", help="How long you want your files to be. Specify a list of two numbers of seconds and the audio files will have durations which fall in that range." \
                                             "For example, '4 8' will get you files whose durations falls between 4 and 8 seconds", type=int, nargs=2)
parser.add_argument("-a", "--audio_directory", help="The directory where your audio files are. Make sure they match the csv", type=str)
parser.add_argument("-c", "--csv_path", help="The path where the csv is located", type=str)
parser.add_argument("-s", "--storage_directory", help="the directory where you want to store the files you asked for", type=str)
args = parser.parse_args()
if __name__ == '__main__':
    number_of_files = args.number_of_files
    characteristics =args.list_of_characteristics
    duration = args.duration
    audio_directory =  args.audio_directory
    csv_path = args.csv_path
    storage_directory = args.storage_directory
    main(csv_path, storage_directory = storage_directory, audio_directory = audio_directory, \
    number_of_files = number_of_files, characteristics = characteristics, duration = duration)
