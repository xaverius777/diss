import re
import os
import pandas as pd
import numpy as np
import emoji
# The noisy and enhanced directories in which CC-Rater will be evaluated
noisy_dir = "/home/jtorres/noisy_enhanced_2/noisy_enhanced/noisy_speech/"
enhanced_dir = "/home/jtorres/noisy_enhanced_2/noisy_enhanced/enhanced_speech/"
# The csv that comes from Qualtrics and the surveys' metafile.
input_csv_path = "C:/Users/javic/PycharmProjects/dissertation/venv/testset_csv.csv"
metafile_path = "C:/Users/javic/PycharmProjects/dissertation/venv/survey_batch_202208102104.txt"
# Output csv
output_csv_path = "C:/Users/javic/Downloads/rome/testset.csv"
# Read the input csv and the metafile
results = pd.read_csv(input_csv_path)
with open(metafile_path) as f:
    metafile = f.readlines()
# Initialize the dictionary in which we will store the list of scores for each audio pair
genders = ['female', 'male']
accents = ['american', 'aussie', 'canadian', 'english', 'indian']
scores_dict = {f"{gender}_{accent}_{number}": [] for gender in genders for accent in accents for number in range(1, 21)}
#Generate the survey substrings that identify which survey a question belongs to (e.g. the "S1" in "S1_QID5"
number_of_surveys= 20
surveys = [f'S{number}' for number in range(1,number_of_surveys + 1)]
answers_dict = {
    "A is much better than B" : -2,
    "A is better than B": -1,
    "I have no preference": 0,
    "B is better than A": 1,
    "B is much better than A": 2
}
filepath_dict = {}
# Set threshold for validation errors allowed, initialize the dictionary that will contain the audios' filenames
val_errors_allowed = 1
# An ad-hoc function to make code more elegant
def get_dict_list(survey_metafile, survey_number):
    """
    :param survey_metafile: the metafile of the survey you're currently processing
    :param survey_number: the number of the survey
    :return: a list of column names for the metafile's questions and a dictionary
    in which each key is a column name (e.g. a question ID) and each value is that question's information
    """
    q_list = []
    q_dict = {}
    for item in survey_metafile:
        qid = (int((item.split(" ")[3]).replace(".", "")))
        column = f"S{survey_number}_QID{qid}"
        q_list.append(column)
        q_dict[column] = item
    return q_list, q_dict
for survey in surveys:
    # Get the survey section from the input csv
    survey_results = results.filter(regex=(fr"{survey}_.*")).dropna()
    survey_results.index = range(1, survey_results.shape[0] + 1)
    if survey_results.empty:
        # Discard empty surveys and inform the user. This doesn't work because it's a single survey
        # and everyone answers consent and instructions, therefore no survey will truly be empty
        print(f"Survey {survey} has not been taken by anyone")
    else:
        print(f" Processing survey {survey}")
        # Get the metafile's section about the survey
        metafile_survey_patt = re.compile(fr"Survey {survey[1:]}:.*")
        metafile_survey = list(filter(metafile_survey_patt.match, metafile))
        # Get the validation and test questins
        test_patt = re.compile(r".*Test question.*")
        val_patt = re.compile(r".*Validation question.*")
        metafile_survey_test_questions = list(filter(test_patt.match, metafile_survey))
        metafile_survey_val_questions = list(filter(val_patt.match, metafile_survey))
        # Get a list of the question IDs and a dictionary question ID: question information in order
        # to process validation and test questions separately
        survey_val_columns, survey_val_dict = get_dict_list( metafile_survey_val_questions, survey[1:])
        survey_test_columns, survey_test_dict = get_dict_list(metafile_survey_test_questions, survey[1:])
        # Get a dataframe for the validation questions and a dataframe for the test questions
        survey_val_data = survey_results[[*survey_val_columns]]
        survey_test_data = survey_results.drop(survey_val_columns, axis=1)
        # parse the survey by iterating over each subject's responses
        for number in range(2, len(survey_results)):
            # get the row
            # print(f"row number {number}")
            val_row = (survey_val_data.iloc[[number]])
            val_row.index = range(val_row.shape[0])
            val_errors = 0
            # iterate over the questions
            for question in survey_val_columns:
                # print((survey_val_dict[question].split(";")[0]))
                result_string = val_row.iloc[0][question]
                result = answers_dict[result_string]
                # result = int(val_row.iloc[0][question])
                # result = answers_dict[val_row.iloc[0][question]]
                if "The audios were inverted" in survey_val_dict[question]:
                    result = result * (-1)
                    # print("The audios had been inverted, so we multiplied the score by -1")
                val_type = survey_val_dict[question].split("'")[1]
                if val_type == "aa":
                    target = 0
                    #print(f"Result: {result}. Target: {target}")
                    if result == target:
                        continue
                        # print(f"Validation question {question}, type {val_type}, was passed")
                    else:
                        val_errors += 1
                        # print(f"Validation question {question}, type {val_type}, was failed")
                if val_type == "a/b":
                    target = [2, 1]
                    #print(f"Result: {result}. Target: {target}")
                    if result in target:
                        continue
                        # print(f"Validation question {question}, type {val_type}, was passed")
                    else:
                        val_errors += 1
                        # print(f"Validation question {question}, type {val_type}, was failed")
            if val_errors > val_errors_allowed:
                print( f"Survey {survey[1:]}, subject {number - 1}: this person failed {val_errors} questions, therefore failed validation. Their test data won't be processed", emoji.emojize(':thumbs_down:'))
                valid_survey = False
            else:
                valid_survey = True
                print(f"Survey {survey[1:]}, subject {number - 1}: this person passed validation! Their test data will be processed", emoji.emojize(':thumbs_up:'))
            # if they passed the validation questions, their test data will be processed
            if valid_survey:
                test_row = (survey_test_data.iloc[[number]])
                for question in test_row:
                    info = survey_test_dict[question].split(";")[0]
                    file = survey_test_dict[question].split(";")[1]
                    speaker = (re.findall(r"(((((fema)|(ma))le)_[a-z]+_(([0-9][0-9])))|((((fema)|(ma))le)_[a-z]+_(([0-9])|([0-9][0-9])?)))", file))[0][0]
                    basename = os.path.basename(file)
                    # try:
                    #     print(filepath_dict[speaker], type(filepath_dict[speaker]))
                    # except KeyError:
                    #     print("filepath dict entry not found for", speaker)
                    filepath_dict[speaker] = basename
                    # print(speaker, basename, file)
                    result_string = test_row.iloc[0][question]
                    result = answers_dict[result_string]
                    # result = int(test_row.iloc[0][question])
                    #result = answers_dict[test_row.iloc[0][question]]
                    if "The audios were inverted" in survey_test_dict[question]:
                        result = result * (-1)
                        # print("The audios had been inverted, so we multiplied the score by -1")
                    scores_dict[speaker].append(result)
            else:
                continue

# Initialize a dataframe in which to store the results
results_csv_column_names = ["A", "B", "CCR_MOS", "no_answers", "answers"]
results_csv = pd.DataFrame(columns = results_csv_column_names)
# Iterate over the audios and their respective scores to populate the dataframe
for audio in scores_dict.keys():
    if len(scores_dict[audio]) >= 1:
        answers = scores_dict[audio]
        filepath = filepath_dict[audio].replace("-", "_")
        noisy_path = noisy_dir + filepath
        enhanced_path = enhanced_dir + filepath
        no_answers = len(scores_dict[audio])
        ccr_scores = np.array(scores_dict[audio])
        ccr_mos = np.mean(ccr_scores)
        # "eureka"
        new_row = {"A": noisy_path, "B": enhanced_path, \
                   "CCR_MOS": ccr_mos, "no_answers": no_answers, "answers": answers}
        new_row_df = pd.DataFrame([new_row])
        results_csv = pd.concat([results_csv, new_row_df], axis=0, ignore_index=True)
# Set the index to start from one for cosmetic reasons
results_csv.index = np.arange(1, len(results_csv.index) + 1)
# Write the dataframe to a csv file
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(results_csv)
results_csv.to_csv(output_csv_path)
