import librosa, soundfile
import re
from pathlib import Path
import shutil
import os
import argparse
def main(parent_dir, target_dir, subdirectories, lists_of_files, common_root):
    parent_dir = parent_dir
    target_dir = target_dir
    common_root = common_root
    with open(lists_of_files) as file:
        lines = file.readlines()
    folder_names = subdirectories
    files_dir = {folder_names[number]:lines[number].split(",") for number in range(len(folder_names))}
    for value in files_dir.values():
        value[-1] = value[-1].strip()
    mode = 0o666
    for key, value in files_dir.items():
        target_path = os.path.join(target_dir, key)
        file_counter = 1
        try:
            os.mkdir(target_path, mode)
            print(f"The subdirectory {target_dir + '/' + key} was created to store the files")
        except FileExistsError:
            print(f"The subdirectory {target_dir + '/' + key} already existed, the files will be stored in it")
        for file in value:
            source_file = parent_dir + "/" + key + "/" + common_root + file + ".mp3"
            shutil.copy(source_file, target_path)
            original_copied_name = target_path + "/" + common_root + file + ".mp3"
            new_name = target_path + "/" + f"{key}_{file_counter}.mp3"
            file_counter +=1
            os.rename(original_copied_name, new_name)
            print(f"file {file} was copied to the folder {target_path}")
        print(f"The files from {parent_dir + '/' + key} were successfully copied to {target_path}!")
parser = argparse.ArgumentParser(description = "Copy files in the subdirectories of a directory to subdirectories of the same name in another directory." \
                                 "Very useful if you just have to copy some of the files, instead of all of them")
parser.add_argument("-p", "--parent_dir", help="The parent directory that contains the subdirectories you want to copy files from", type=str)
parser.add_argument("-t", "--target_dir", help="The target directory in which you'll create subdirectories identical as those in the parent dict." \
                          "The files in the 'folder_files' argument will be stored in this directory.", type=str)
parser.add_argument("-s", "--subdirectories" , help="The subdirectories you are copying the files from. Subdirectories with an identical name" \
                                             "will be created in the 'target_dir' directory, if they didn't exist yet in said directory." \
                                             "Furthermore, every file copied to each subdirectory will be renamed to the name of the subdirectory" \
                                             "plus what number of copied file it is. For example, if we copy files tencatron43, pentatron12 and ocmalon86" \
                                             "to subdirectory robot, they'll be renamed to robot_1, robot_2, robot_3"  , type=str, nargs= '+')
parser.add_argument("-l", "--lists_of_files", help="A txt file in which every line is a list of the filenames you want to get from each subdirectory, separated by commas." \
                    "For example, a line: tencatron43,pentatron12,ocmalon86. There must be as many lines as subdirectories, and they must be aligned" \
                    "for the program to work", type=str)
parser.add_argument("-c", "--common_root", help="In case all your files share a common root, specify it here. Instead of adding the root" \
                                                "to every filename in the lists_of_files argument, you can specify this argument and the program" \
                                                "will do it for you", type=str, nargs='?')
args = parser.parse_args()
if __name__ == '__main__':
    parent_dir = args.parent_dir
    target_dir = args.target_dir
    subdirectories = args.subdirectories
    lists_of_files = args.lists_of_files
    common_root = args.common_root
    main(parent_dir = parent_dir , target_dir = target_dir, subdirectories = subdirectories, \
         lists_of_files = lists_of_files, common_root = common_root)
