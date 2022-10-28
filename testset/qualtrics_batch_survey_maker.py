import argparse
import json
import copy
from string import Template
from collections import OrderedDict
import os
import sys
import config_v2
import random
import datetime
# local imports
json_filename = "C:/Users/javic/PycharmProjects/dissertation/venv/ccr_multiple_choice.json"
audio_html_template = "C:/Users/javic/PycharmProjects/qualtreats/audio_template.html"
play_button = "C:/Users/javic/PycharmProjects/dissertation/venv/play_button.html"
save_as = "C:/Users/javic/Downloads/rome/ccr_mc.qsf"
# load JSON template from file
def get_basis_json():
    with open(json_filename, encoding='utf8') as json_file:
        return json.load(json_file)
# standard audio player for all question types except MUSHRA

def get_player_html(url):
    with open(audio_html_template) as html_file:
        return Template(html_file.read()).substitute(url=url)
# audio player with only play/pause controls for MUSHRA tests
# to prevent participants identifying hidden reference by duration
def get_play_button(url, n): # player n associates play button with a specific audio
    with open(play_button) as html_file:
        return Template(html_file.read()).substitute(url=url, player=n)
# Function to set the ID of objects
def set_id(obj):
    obj['SurveyID'] = config_v2.survey_id
    return obj
# Function to make a new question
def make_question(qid, urls, basis_question,question_type,
                  question_function, question_text, slice_counter):
    new_q = copy.deepcopy(basis_question)
    # Set the survey ID
    new_q['SurveyID'] = config_v2.survey_id
    # Change all the things that reflect the question ID
    new_q['Payload'].update({'QuestionID' : f'QID{qid}',
                             #DataExportTag is the name by which each question shows up in the results csv
                                   'DataExportTag' : f'S{slice_counter}_QID{qid}',
                                   'QuestionDescription' : f'Q{qid}:{question_type}',
                                   'QuestionText': question_text})
    # The change from "QID(question id)" to "S(survey id)_QID(question id)" makes results easier to parse
    new_q.update({'PrimaryAttribute' : f'QID{qid}',
                        'SecondaryAttribute' : f'QID{qid}: {question_type}' })
    try: # call handler function for each question type
        question_function(new_q, urls, qid)
    except TypeError:
        pass
    return new_q
# make n new blocks according to the survey_length (I don't really understand this)
def make_blocks(num_questions, basis_blocks):
    new_blocks = basis_blocks
    block_elements = []
    for i in range(1,num_questions+1):
        block_element = OrderedDict()
        block_element['Type'] = 'Question'
        block_element['QuestionID'] = f'QID{i}'
        block_elements.append(block_element)
    new_blocks['Payload']['2']['BlockElements'] = block_elements
    return new_blocks
# Files necessary to conduct a CCR test, including validation
noisy = "C:/Users/javic/PycharmProjects/dissertation/venv/noisy_speech_urls.txt"
#Val noisy should be the same noisy-enhanced pairs, but there is no 15 or 25 snr.
val_noisy = "C:/Users/javic/PycharmProjects/dissertation/venv/val_noisy_speech_urls.txt"
enhanced = "C:/Users/javic/PycharmProjects/dissertation/venv/enhanced_speech_urls.txt"
clean = "C:/Users/javic/PycharmProjects/dissertation/venv/clean_speech_urls.txt"
# Make the tuples of noisy speech and enhanced speech.
# To make the surveys, get slices out of this list. The size of the slices will be a parameter.
with open(noisy) as f1:
    with open(enhanced) as f2:
        test_list = [(line1.replace('\n', ' ').replace('\r', ''), line2.replace('\n', ' ').replace('\r', '')) for line1, line2 in zip(f1,f2)]
with open(val_noisy) as f3:
    with open(clean) as f4:
        validation_list = [(line1.replace('\n', ' ').replace('\r', ''), line2.replace('\n', ' ').replace('\r', '')) for line1, line2 in zip(f3,f4)]

def test_question(url_pair, qid):
    coin_toss = random.choice([True, False])
    if coin_toss:
        inverse_message = ""
        url_1 = url_pair[0]
        url_2 = url_pair[1]
    else:
        inverse_message = "The audios were inverted."
        url_1 = url_pair[1]
        url_2 = url_pair[0]
    html_blankspace_string = "&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp;"
    question_text = f"Question {qid}: {config_v2.mos_question_text}<br><br>" \
                    f"A&nbsp;{get_player_html(url_1)}{html_blankspace_string}" \
                    f"B&nbsp;{get_player_html(url_2)}"
    qid = qid
    new_question = make_question(qid, url_pair, \
                                 basis_question, question_type, question_function, question_text, slice_counter)
    # hehe
    elements[-1]['Payload']['Validation']['Settings']['ForceResponse'] = "ON"
    message = f"Survey {slice_counter}: question {qid}. Test question. " + inverse_message
    print(message)
    log_addition = message + ";" + url_1 + ";" + url_2
    session_log.append(log_addition)
    qid += 1
    return new_question, qid
def val_question(url_pair, qid, mode=str):
    # retrieve the validation pair correspondent for the current test pair
    # val_url_pair = [url_pair for url_pair]
    coin_toss = random.choice([True, False])
    if mode == "aabb":
        mode_message = f"Validation question, type 'aa'."
        inverse_message = ""
        url_1 = url_pair[0]
        url_2 = url_pair[0]

    else:
        mode_message = f"Validation question, type 'a/b'. "
        if coin_toss:
            inverse_message = ""
            url_1 = url_pair[0]
            url_2 = url_pair[1]
        else:
            inverse_message = "The audios were inverted. "
            url_1 = url_pair[1]
            url_2 = url_pair[0]
    html_blankspace_string = "&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp;"
    question_text = f"Question {qid}: {config_v2.mos_question_text}<br><br>" \
                    f"A&nbsp;{get_player_html(url_1)}{html_blankspace_string}" \
                    f"B&nbsp;{get_player_html(url_2)}"
    qid = qid
    new_question = make_question(qid, url_pair, \
                                 basis_question, question_type, question_function, question_text, slice_counter)
    # hehe
    elements[-1]['Payload']['Validation']['Settings']['ForceResponse'] = "ON"
    message = f"Survey {slice_counter}: question {qid}. " + mode_message + inverse_message
    print(message)
    log_addition = message + ";" + url_1 + ";" + url_2
    session_log.append(log_addition)
    qid += 1
    return new_question, qid

# Shuffle the lists so the surveys won't be predictable
session_log = [str(datetime.datetime.now())]
date_n_time = ''.join(ch for ch in (str(datetime.datetime.now())[0:16]) if ch.isalnum())
random.shuffle(test_list)
random.shuffle(validation_list)
test_questions = 10
val_questions_aabb = 3
val_questions_ab = 3
# val_checker works as longs as the number of validation files doesn't exceed the number of test files
val_aabb = ["aabb" for number in range(val_questions_aabb)]
val_ab = ["ab" for number in range(val_questions_ab)]
val_checker = val_aabb + val_ab
for number in range((test_questions - (len(val_aabb) + len(val_ab)))):
    val_checker.append("test")
slice_start = 0
slice_counter = 1
question_type = "mos"
question_function = "None"
while (slice_start + test_questions) <= len(test_list):
    slice = test_list[slice_start: slice_start + test_questions]
    slice_start += test_questions
    output_path = save_as.replace(".qsf", f"_{slice_counter}.qsf")
    basis_json = get_basis_json()
    elements = basis_json['SurveyElements']
    elements = list(map(set_id, elements))
    basis_question = elements[-1]
    qid = 1
    questions = []
    question_files = [url_pair[0] for url_pair in slice]
    validation_pool = validation_list[:]
    for file in question_files:
        for audio_pair in validation_pool:
            if audio_pair[0] == file:
                validation_pool.remove(audio_pair)
    for file in question_files:
        assert file not in validation_pool
    random.shuffle(validation_pool)
    survey_val_checker = val_checker[:]
    random.shuffle(survey_val_checker)
    # Iterate over the file pairs to make the questions
    for url_pair in slice:
        what_do = survey_val_checker.pop()
        # print(val_checker)
        if what_do == "aabb":
            val_pair = validation_pool.pop()
            val_q, qid = val_question(val_pair, qid, mode="aabb")
            questions.append(val_q)
            new_question, qid = test_question(url_pair, qid)
            questions.append(new_question)
        elif what_do == "ab":
            # print("Validation question of the type ab")
            val_pair = validation_pool.pop()
            val_q, qid = val_question(val_pair, qid, mode="ab")
            questions.append(val_q)
            new_question, qid = test_question(url_pair, qid)
            questions.append(new_question)
        else:
            new_question, qid = test_question(url_pair, qid)
            questions.append(new_question)

    basis_blocks = elements[0]
    basis_flow = elements[1]
    rs = elements[2]
    basis_survey_count = elements[7]
    survey_length = len(questions)
    blocks = make_blocks(survey_length, basis_blocks)
    flow = basis_flow
    flow['Payload']['Properties']['Count'] = survey_length
    survey_count = basis_survey_count
    survey_count["SecondaryAttribute"] = str(survey_length)
    elements = [blocks, flow] + elements[2:7] + questions + [rs]
    out_json = basis_json
    out_json['SurveyElements'] = elements
    # Uncomment next lines when ready to use the program. Like a gun's trigger lock
    with open(output_path, 'w+') as outfile:
        json.dump(out_json, outfile, indent=4)
        print(f"Survey number {slice_counter} finished, wrote it to a file")
    slice_counter += 1
session_log_file = f"C:/Users/javic/Downloads/rome/survey_batch_{date_n_time}.txt"
with open(session_log_file, 'w') as file:
    file.write('\n'.join(session_log))