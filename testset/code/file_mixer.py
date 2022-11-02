import os
import random
import soundfile as sf
import numpy as np

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