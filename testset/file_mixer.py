import os
import random
import soundfile as sf
import numpy as np

args_dict = {'cfg': {'sampling_rate': '16000',
                     'audioformat': '*.wav',
                     'audio_length': '9.99',
                     'silence_length': '0.0',
                     'total_hours': '1.37',
                     'snr_lower': '0',
                     'snr_upper': '40',
                     'randomize_snr': 'True',
                     'target_level_lower': '-35',
                     'target_level_upper': '-15',
                     'total_snrlevels': '5',
                     'clean_activity_threshold': '0.6',
                     'noise_activity_threshold': '0.0',
                     'fileindex_start': '1',
                     'fileindex_end': '2',
                     'is_test_set': 'False',
                     'noise_dir': '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s',
                     'speech_dir': '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/',
                     'noise_types_excluded': 'None',
                     'noisy_destination': './noisy_speech',
                     'clean_destination': './clean_speech',
                     'noise_destination': '/disk/scratch/noise_destination',
                     'log_dir': '/disk/scratch/log_dir',
                     'use_singing_data': '0',
                     'clean_singing': 'datasets\\clean\\singing_voice',
                    'singing_choice': '3',
                     'use_emotion_data': '0',
                     'clean_emotion': 'datasets\\clean\\emotional_speech',
                     'use_mandarin_data': '0',
                     'clean_mandarin': 'datasets\\clean\\mandarin_speech',
                     'rir_choice': '1',
                     'lower_t60': '2',
                     'upper_t60': '1.3',
                     'rir_table_csv': 'datasets\\acoustic_params\\RIR_table_simple.csv',
                     'clean_speech_t60_csv': 'datasets\\acoustic_params\\cleanspeech_table_t60_c50.csv',
                     'snr_test': 'True',
                     'norm_test': 'True',
                     'sampling_rate_test': 'True',
                     'clipping_test': 'True',
                     'unit_tests_log_dir': 'unittests_logs'},
              'fs': 16000,
             'audioformat': '*.wav',
             'audio_length': 9.99,
             'silence_length': 0.0,
             'total_hours': 1.37,
             'use_singing_data': 0,
             'clean_singing': 'datasets\\clean\\singing_voice',
             'singing_choice': 3,
             'use_emotion_data': 0,
             'clean_emotion': 'datasets\\clean\\emotional_speech',
             'use_mandarin_data': 0, 'clean_mandarin': 'datasets\\clean\\mandarin_speech',
             'num_files': 1,
             'fileindex_start': 1,
             'fileindex_end': 2,
             'is_test_set': False,
             'clean_activity_threshold': 0.6,
             'noise_activity_threshold': 0.0,
             'snr_lower': 0,
             'snr_upper': 40,
             'randomize_snr': True,
             'target_level_lower': -35,
             'target_level_upper': -15,
             'snr': 20,
             'noisyspeech_dir': './noisy_speech',
             'clean_proc_dir': './clean_speech',
             'noise_proc_dir': '/disk/scratch/noise_destination',
             'cleanfilenames': ['/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_7.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_8.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_12.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_13.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_15.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_15.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_16.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_18.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_19.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_16.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_20.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_14.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_10.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_17.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_10.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_11.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_14.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_18.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_8.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_6.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_15.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_8.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_18.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_20.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_7.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_11.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_13.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_7.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_19.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_17.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_14.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_12.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_15.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_19.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_6.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_18.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_6.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_11.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_20.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_12.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_19.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_10.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_11.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_19.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_6.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_8.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_19.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_11.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_6.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_7.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_16.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_10.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_11.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_9.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_12.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_9.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_15.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_19.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_18.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_13.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_19.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_13.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_9.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_20.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_9.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_14.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_13.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_17.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_19.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_15.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_12.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_8.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_14.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_9.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_20.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_6.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_15.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_13.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_17.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_7.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_11.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_17.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_6.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_18.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_20.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_10.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_10.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_12.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_20.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_18.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_17.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_14.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_11.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_12.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_18.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_14.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_14.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_20.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_15.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_13.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_17.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_13.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_8.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_9.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_11.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_16.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_12.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_7.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_14.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_6.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_17.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_7.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_7.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_16.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_15.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_6.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_8.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_8.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_10.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_18.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_12.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_11.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_10.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_17.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_8.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_8.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_16.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_9.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_16.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_20.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_14.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_13.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_12.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_16.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_16.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_9.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_7.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_10.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_aussie_18.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_canadian_10.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_13.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_american_9.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_american_17.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_aussie_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_16.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_7.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_english_6.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_indian_20.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_indian_19.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/female_english_15.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/read_speech_s/male_canadian_9.wav'],
             'num_cleanfiles': 200,
             'noisefilenames': ['/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/baby_crying_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/cats_meowing_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/344_audio_AMBCnst_Machinery_Drilling_Harsh_Metal_Power_Tools_On_and_Off_Indoors_344_Audio_UK_Residential_and_Industrial_1690.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/yfs_excavators_contruction_site_idles_and_drives_03_9624_191.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_heavy_breathing_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/car_horns_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_drinking_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/water_pouring_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_drinking_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/pm_gl_train_renfe_spain_idle_ac.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/chairs_creaking_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/human_male_snoring_001.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/packet_of_crisps_opening_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/human_female_snoring.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/can_opening_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/plate_noise_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/doors_opening_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/ftus_household_room_tone_bedroom_fan_in_bg.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/dogs_barking_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/can_opening_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/human_male_snoring_002.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/plate_noise_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/baby_crying_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/car_horns_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/344_audio_ext_iceland_reykjavik_city_construction_road_368.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/household_bathroom_extractor_fan_on_operate_off.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/zapsplat_industrial_airconditioning_unit_fan_exterior_hum_84762.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/water_running_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/zapsplat_household_washing_machine_modern_spin_cycle_001_43569.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/fork_and_plate_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/baby_crying_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/packet_of_crisps_opening_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/tom_chapman_office_room_tone_computer_noise_airy_air_conditioning.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_munching_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/car_horns_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/can_opening_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/344_audio_AMBCnst_Machinery_Saw_Metal_Sawing_Power_Tools_On_and_Off_Indoors_Edited_344_Audio_UK_Residential_and_Industrial_1691.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/zapsplat_transport_car_internal_fan_blower_run_long_off_14793.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/can_opening_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/weather_rain_heavy_with_drips.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/car_horns_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/doors_opening_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/felix_blume_household_washing_machine_cycle_close_recording.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/industrial_air_curtain_machine_on_run_long_off.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/cats_meowing_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/fork_and_plate_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/water_pouring_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/water_pouring_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/water_running_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/zapsplat_nature_thunderstorm_aftermath_light_rain_drips_distant_rumbles_thunder_56659.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/can_opening_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/chairs_creaking_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/water_running_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_heavy_breathing_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_heavy_breathing_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/baby_crying_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/zapsplat_nature_rain_approaching_thunderstorm.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_drinking_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/yfs_excavators_contruction_site_idles_02_9624_187.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/fork_and_plate_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/human_woman_snoring_37_years_old.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/chairs_creaking_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/fork_and_plate_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/chairs_creaking_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/cats_meowing_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/car_horns_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_heavy_breathing_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/door_audio_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/zapsplat_nature_rain_light_metal_roof_thunderstorm_approach.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/packet_of_crisps_opening_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/household_washing_machine_operating.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_munching_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/fork_and_plate_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/water_running_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/zapsplat_human_man_elderly_snore_20087.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/door_audio_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/door_audio_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_drinking_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/dogs_barking_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/packet_of_crisps_opening_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/water_pouring_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/doors_opening_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/dogs_barking_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_heavy_breathing_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/packet_of_crisps_opening_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/PM_Train_Renfe_Spain_Idle_AC_342.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_munching_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/chairs_creaking_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/plate_noise_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/door_audio_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/doors_opening_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/water_pouring_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/water_running_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/dogs_barking_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_drinking_2.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/dogs_barking_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/plate_noise_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/zapsplat_household_washing_machine_modern_spin_cycle_002_43570.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_munching_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/door_audio_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/doors_opening_3.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/zapsplat_nature_ambience_approaching_thunderstorm_rain_wind_intensifies_blustery_rumbles_thunder_74166.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/baby_crying_1.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/plate_noise_4.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/people_munching_5.wav', '/afs/inf.ed.ac.uk/user/s22/s2259600/dissertation_data/noise_s/pm_wshr_22_washer_washing_machine_spin_fast_seamless_loop_294.wav']
             }
def audioread(path, norm=False, start=0, stop=None, target_level=-25):
    '''Function to read audio'''

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        audio, sample_rate = sf.read(path, start=start, stop=stop)
    except RuntimeError:  # fix for sph pcm-embedded shortened v2
        print('WARNING: Audio type not supported')
        return (None, None)

    if len(audio.shape) == 1:  # mono
        if norm:
            rms = (audio ** 2).mean() ** 0.5
            scalar = 10 ** (target_level / 20) / (rms+EPS)
            audio = audio * scalar
    else:  # multi-channel
        audio = audio.T
        audio = audio.sum(axis=0)/audio.shape[0]
        if norm:
            audio = normalize(audio, target_level)

    return audio, sample_rate

def normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    rms = (audio ** 2).mean() ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio

def is_clipped(audio, clipping_threshold=0.99):
    return any(abs(audio) > clipping_threshold)
def snr_mixer(params, clean, noise, snr, target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various SNR levels'''
    # cfg = params['cfg']
    if len(clean) > len(noise):
        noise = np.append(noise, np.zeros(len(clean) - len(noise)))
    else:
        clean = np.append(clean, np.zeros(len(noise) - len(clean)))

    # Normalizing to -25 dB FS
    clean = clean / (max(abs(clean)) + EPS)
    clean = normalize(clean, target_level)
    rmsclean = (clean ** 2).mean() ** 0.5

    noise = noise / (max(abs(noise)) + EPS)
    noise = normalize(noise, target_level)
    rmsnoise = (noise ** 2).mean() ** 0.5

    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10 ** (snr / 20)) / (rmsnoise + EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel

    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue.
    noisy_rms_level = np.random.randint(params['target_level_lower'], params['target_level_upper'])
    rmsnoisy = (noisyspeech ** 2).mean() ** 0.5
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy + EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy

    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech):
        noisyspeech_maxamplevel = max(abs(noisyspeech)) / (clipping_threshold - EPS)
        noisyspeech = noisyspeech / noisyspeech_maxamplevel
        clean = clean / noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel / noisyspeech_maxamplevel
        noisy_rms_level = int(20 * np.log10(scalarnoisy / noisyspeech_maxamplevel * (rmsnoisy + EPS)))

    return clean, noisenewlevel, noisyspeech, noisy_rms_level
def audiowrite(destpath, audio, sample_rate=16000, norm=False, target_level=-25, \
                clipping_threshold=0.99, clip_test=False):
    '''Function to write audio'''

    if clip_test:
        if is_clipped(audio, clipping_threshold=clipping_threshold):
            raise ValueError("Clipping detected in audiowrite()! " + \
                            destpath + " file not written to disk.")

    if norm:
        audio = normalize(audio, target_level)
        max_amp = max(abs(audio))
        if max_amp >= clipping_threshold:
            audio = audio/max_amp * (clipping_threshold-EPS)

    destpath = os.path.abspath(destpath)
    destdir = os.path.dirname(destpath)

    if not os.path.exists(destdir):
        os.makedirs(destdir)

    sf.write(destpath, audio, sample_rate)
    return

storage_dir = "folder where you want to store your noisy speech files"
clean_input_dir="folder where your clean speech files are stored"
noise_input_dir="folder where your noisy speech files are stored"
clean_filename_list = [file for file in os.listdir(clean_input_dir)]
noise_filename_list = [file for file in os.listdir(noise_input_dir)]
initialize = np.zeros(0)
params = {
    'target_level_lower': -35,
    'target_level_upper': -15
}
EPS = np.finfo(float).eps
for number in range(len(clean_filename_list)):
    # Test snr
    #snr = random.choice([-5, 0, 5, 15, 25])
    # Val snr. Remove two highest values so noisy files will always be
    # nitidly noisier than their clean counterpoint
    snr = random.choice([-5, 0, 5])
    clean_path = clean_input_dir + clean_filename_list[number]
    clean_input_audio, clean_fs_input = audioread(clean_path)
    clean_audio = np.append(initialize, clean_input_audio)
    noise_filename = random.choice(noise_filename_list)
    print(f"Mixing clean speech file {clean_filename_list[number]} with noise file {noise_filename}...")
    noise_path = noise_input_dir + noise_filename
    noise_input_audio, noise_fs_input = audioread(noise_path)
    noise_audio = np.append(initialize, noise_input_audio)
    clean_length, noise_length = clean_audio.shape[0], noise_audio.shape[0]
    slice_start = random.randint(0, noise_length - clean_length)
    noise_audio = noise_audio[slice_start:slice_start+clean_length]
    clean_speech, noise_level, noisy_speech, noisy_rms_level = \
        snr_mixer(params, clean_audio, noise_audio, snr, target_level=-25, clipping_threshold=0.99)
    noisy_filename = clean_filename_list[number].replace(".wav", "") + "_" + noise_filename.replace(".wav", "") + f"_snr_{snr}_rms_{noisy_rms_level}.wav"
    storage_path = storage_dir + noisy_filename
    audiowrite(storage_path, noisy_speech, sample_rate=16000, norm=False, target_level=-25, \
                     clipping_threshold=0.99, clip_test=False)
    print(f"Noisy speech file {noisy_filename} was born. It has been put in directory {storage_dir}")