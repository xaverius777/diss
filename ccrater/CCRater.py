import random
import os
import re
import argparse
import concurrent.futures
import glob
import os
import librosa
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import soundfile as sf
from requests import session
from tqdm import tqdm
import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import copy
import gc
import time

#In this directory are located the variables folder, which DNSMOS uses to run,
#and the files in which the model checkpoints are saved.
ROOT_DIR = "/home/your_dir/"


# CC-RATER MODEL. COMPOSED BY:
# DNSMOS BLOCK
# CLASSIFIER BLOCK
# CC-RATER BLOCK, WHICH PUTS TWO PREVIOUS BLOCKS TOGETHER

class DNSMOS(nn.Module):
    '''
    DNSMOS BLOCK
    Input: Audio windows. Shape: [Batch size, number of windows, window size]
    Output: Acoustic features. Shape: [Batch size, number of windows, 64, 112, 20]
    '''
    def __init__(self):
        super(DNSMOS, self).__init__()
        self._vars = nn.ParameterDict()
        self._regularizer_params = []
        #These variables are defined separately, as they were giving trouble.
        #Don't include them in the variables folder
        self._vars["mos_estimator_logpow_Maximum_x_0"] = nn.Parameter(torch.tensor([9.999999960041972e-13], dtype = torch.float32), requires_grad=True)
        self._vars["mos_estimator_logpow_pow_y_0"] = nn.Parameter(torch.tensor([2.0]), requires_grad=True)
        self._vars["mos_estimator_logpow_truediv_y_0"] = nn.Parameter(torch.tensor([2.3025851249694824]), requires_grad=True)
        for b in glob.glob(os.path.join(ROOT_DIR, "variables", "*.npy")):
            v = torch.from_numpy(np.load(b))
            requires_grad = v.dtype.is_floating_point or v.dtype.is_complex
            self._vars[os.path.basename(b)[:-4]] = nn.Parameter(v, requires_grad=requires_grad)
        self.n_stft_real = nn.Conv1d(**{'groups': 1, 'dilation': [1], 'out_channels': 161, 'padding': 0, 'kernel_size': (1,), 'stride': [1], 'in_channels': 320, 'bias': False})
        self.n_stft_real.weight.data = self._vars["time2freq_stft-real_kernel_0"]
        self.n_stft_imag = nn.Conv1d(**{'groups': 1, 'dilation': [1], 'out_channels': 161, 'padding': 0, 'kernel_size': (1,), 'stride': [1], 'in_channels': 320, 'bias': False})
        self.n_stft_imag.weight.data = self._vars["time2freq_stft-imag_kernel_0"]
        self.n_conv2d = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 128, 'padding': [1, 1], 'kernel_size': (3, 3), 'stride': [1, 1], 'in_channels': 1, 'bias': True})
        self.n_conv2d.weight.data = self._vars["conv2d_kernel_0"]
        self.n_conv2d.bias.data = self._vars["conv2d_bias_0"]
        self.n_conv2d_1 = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 64, 'padding': [1, 1], 'kernel_size': (3, 3), 'stride': [1, 1], 'in_channels': 128, 'bias': True})
        self.n_conv2d_1.weight.data = self._vars["conv2d_1_kernel_0"]
        self.n_conv2d_1.bias.data = self._vars["conv2d_1_bias_0"]
        self.n_conv2d_2 = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 64, 'padding': [1, 1], 'kernel_size': (3, 3), 'stride': [1, 1], 'in_channels': 64, 'bias': True})
        self.n_conv2d_2.weight.data = self._vars["conv2d_2_kernel_0"]
        self.n_conv2d_2.bias.data = self._vars["conv2d_2_bias_0"]
        self.n_conv2d_3 = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 32, 'padding': [1, 1], 'kernel_size': (3, 3), 'stride': [1, 1], 'in_channels': 64, 'bias': True})
        self.n_conv2d_3.weight.data = self._vars["conv2d_3_kernel_0"]
        self.n_conv2d_3.bias.data = self._vars["conv2d_3_bias_0"]
        self.n_mos_estimator_logpow_conv2d_3_Relu_0_pooling = nn.MaxPool2d(**{'dilation': 1, 'kernel_size': [2, 2], 'ceil_mode': False, 'stride': [2, 2], 'return_indices': False})
        self.n_conv2d_4 = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 32, 'padding': [1, 1], 'kernel_size': (3, 3), 'stride': [1, 1], 'in_channels': 32, 'bias': True})
        self.n_conv2d_4.weight.data = self._vars["conv2d_4_kernel_0"]
        self.n_conv2d_4.bias.data = self._vars["conv2d_4_bias_0"]
        self.n_mos_estimator_logpow_max_pooling2d_MaxPool_1_conv = nn.MaxPool2d(**{'dilation': 1, 'kernel_size': [2, 2], 'ceil_mode': False, 'stride': [2, 2], 'return_indices': False})
        self.n_conv2d_5 = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 32, 'padding': [1, 1], 'kernel_size': (3, 3), 'stride': [1, 1], 'in_channels': 32, 'bias': True})
        self.n_conv2d_5.weight.data = self._vars["conv2d_5_kernel_0"]
        self.n_conv2d_5.bias.data = self._vars["conv2d_5_bias_0"]
        self.n_mos_estimator_logpow_max_pooling2d_MaxPool_2_conv = nn.MaxPool2d(**{'dilation': 1, 'kernel_size': [2, 2], 'ceil_mode': False, 'stride': [2, 2], 'return_indices': False})
        self.n_conv2d_6 = nn.Conv2d(**{'groups': 1, 'dilation': [1, 1], 'out_channels': 64, 'padding': [1, 1], 'kernel_size': (3, 3), 'stride': [1, 1], 'in_channels': 32, 'bias': True})
        self.n_conv2d_6.weight.data = self._vars["conv2d_6_kernel_0"]
        self.n_conv2d_6.bias.data = self._vars["conv2d_6_bias_0"]
    def forward(self, *inputs):
        input_1, = inputs
        #print("DNSMOS INPUT", type(input_1), input_1.shape, input_1.requires_grad)
        original_input = input_1[:, :, :]
        input_1 = torch.reshape(input_1, [-1, 144160])
        input_1_01_cropping0 = input_1[:, :144000]
        input_1_01_cropping01 = input_1[:, 160:]
        mos_estimator_logpow_Reshape_0 = torch.reshape(input_1_01_cropping0, [s if s != 0 else input_1_01_cropping0.shape[i] for i, s in enumerate(self._vars["shape_tensor"])])
        mos_estimator_logpow_Reshape_1_0 = torch.reshape(input_1_01_cropping01, [s if s != 0 else input_1_01_cropping01.shape[i] for i, s in enumerate(self._vars["shape_tensor1"])])
        mos_estimator_logpow_concat_0 = torch.cat((mos_estimator_logpow_Reshape_0, mos_estimator_logpow_Reshape_1_0), **{'dim': 2})
        adjusted_input7 = mos_estimator_logpow_concat_0.permute(*[0, 2, 1])
        adjusted_input8 = mos_estimator_logpow_concat_0.permute(*[0, 2, 1])
        adjusted_input7 = self.compatible_auto_pad(adjusted_input7, self.n_stft_real.weight.data.shape[2:], self.n_stft_real, 'VALID')
        convolution_output7 = self.n_stft_real(adjusted_input7)
        adjusted_input8 = self.compatible_auto_pad(adjusted_input8, self.n_stft_imag.weight.data.shape[2:], self.n_stft_imag, 'VALID')
        convolution_output8 = self.n_stft_imag(adjusted_input8)
        transpose_output7 = convolution_output7.permute(*[0, 2, 1])
        transpose_output8 = convolution_output8.permute(*[0, 2, 1])
        mos_estimator_logpow_time2freq_Square_0 = torch.mul(transpose_output7, transpose_output7)
        mos_estimator_logpow_time2freq_Square_1_0 = torch.mul(transpose_output8, transpose_output8)
        mos_estimator_logpow_time2freq_Add_0 = torch.add(mos_estimator_logpow_time2freq_Square_0, mos_estimator_logpow_time2freq_Square_1_0)
        mos_estimator_logpow_time2freq_Sqrt_0 = torch.sqrt(mos_estimator_logpow_time2freq_Add_0)
        mos_estimator_logpow_pow_0 = torch.pow(mos_estimator_logpow_time2freq_Sqrt_0, self._vars["mos_estimator_logpow_pow_y_0"])
        mos_estimator_logpow_Maximum_max0 = torch.max(self._vars["mos_estimator_logpow_Maximum_x_0"], mos_estimator_logpow_pow_0)
        mos_estimator_logpow_Log_0 = torch.log(mos_estimator_logpow_Maximum_max0)
        mos_estimator_logpow_truediv_0 = torch.div(mos_estimator_logpow_Log_0, self._vars["mos_estimator_logpow_truediv_y_0"])
        mos_estimator_logpow_ExpandDims_0 = torch.unsqueeze(mos_estimator_logpow_truediv_0, 3)
        adjusted_input6 = mos_estimator_logpow_ExpandDims_0.permute(*[0, 3, 1, 2])
        convolution_output6 = self.n_conv2d(adjusted_input6)
        mos_estimator_logpow_conv2d_Relu_0 = F.relu(convolution_output6)
        convolution_output5 = self.n_conv2d_1(mos_estimator_logpow_conv2d_Relu_0)
        mos_estimator_logpow_conv2d_1_Relu_0 = F.relu(convolution_output5)
        convolution_output4 = self.n_conv2d_2(mos_estimator_logpow_conv2d_1_Relu_0)
        mos_estimator_logpow_conv2d_2_Relu_0 = F.relu(convolution_output4)
        convolution_output3 = self.n_conv2d_3(mos_estimator_logpow_conv2d_2_Relu_0)
        mos_estimator_logpow_conv2d_3_Relu_0 = F.relu(convolution_output3)
        mos_estimator_logpow_conv2d_3_Relu_0_pooling0 = self.n_mos_estimator_logpow_conv2d_3_Relu_0_pooling(mos_estimator_logpow_conv2d_3_Relu_0)
        convolution_output2 = self.n_conv2d_4(mos_estimator_logpow_conv2d_3_Relu_0_pooling0)
        mos_estimator_logpow_conv2d_4_Relu_0 = F.relu(convolution_output2)
        mos_estimator_logpow_max_pooling2d_MaxPool_1_conv0 = self.n_mos_estimator_logpow_max_pooling2d_MaxPool_1_conv(mos_estimator_logpow_conv2d_4_Relu_0)
        convolution_output1 = self.n_conv2d_5(mos_estimator_logpow_max_pooling2d_MaxPool_1_conv0)
        mos_estimator_logpow_conv2d_5_Relu_0 = F.relu(convolution_output1)
        mos_estimator_logpow_max_pooling2d_MaxPool_2_conv0 = self.n_mos_estimator_logpow_max_pooling2d_MaxPool_2_conv(mos_estimator_logpow_conv2d_5_Relu_0)
        convolution_output = self.n_conv2d_6(mos_estimator_logpow_max_pooling2d_MaxPool_2_conv0)
        mos_estimator_logpow_conv2d_6_Relu_0 = F.relu(convolution_output)
        features = torch.reshape(mos_estimator_logpow_conv2d_6_Relu_0, [original_input.shape[0], original_input.shape[1], 64, 112, 20])
        return features
    def compatible_auto_pad(self, input, kernel_spatial_shape, nn_mod, auto_pad=None, **kwargs):
        input_spatial_shape = input.shape[2:]
        d = len(input_spatial_shape)
        strides = nn_mod.stride
        dilations = nn_mod.dilation
        output_spatial_shape = [math.ceil(float(l) / float(r)) for l, r in zip(input.shape[2:], strides)]
        pt_padding = [0] * 2 * d
        pad_shape = [0] * d
        for i in range(d):
            pad_shape[i] = (output_spatial_shape[i] - 1) * strides[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
            mean = pad_shape[i] // 2
            if auto_pad == b"SAME_UPPER":
                l, r = pad_shape[i] - mean, mean
            else:
                l, r = mean, pad_shape[i] - mean
            pt_padding.insert(0, r)
            pt_padding.insert(0, l)
        return F.pad(input, pt_padding)

class classifier(nn.Module):
    '''
    CLASSIFIER BLOCK
    Input:
    - Audio embeddings A. Shape: [Batch size, number of windows, 64, 112, 20]
    - Audio embeddings B. Shape: [Batch size, number of windows, 64, 112, 20]
    -Output: CCR score. Shape: [Batch size, 1]
    '''
    def __init__(self):
        super(classifier, self).__init__()
        self.conv_2d_1_1_k = nn.Conv2d(kernel_size=(1, 1),
                                       padding=0,
                                       in_channels=128,
                                       out_channels=128)
        self.linear_128_256 = nn.Linear(128, 256)
        self.linear_256_128 = nn.Linear(256, 128)
        self.linear_128_1 = nn.Linear(128, 1)

    def forward(self, feat_1, feat_2):
        features_1, features_2 = feat_1, feat_2
        concatenation = torch.cat((features_1, features_2), 2)
        concatenation = torch.reshape(concatenation, [-1, concatenation.shape[2], concatenation.shape[3], concatenation.shape[4]])
        convolved = self.conv_2d_1_1_k(concatenation)
        max_global_pool = torch.amax(convolved, dim=(2, 3), keepdim=False)
        dl_1 = self.linear_128_256(max_global_pool)
        dl_1_relued = F.relu(dl_1, inplace=False)
        dl_2 = self.linear_256_128(dl_1_relued)
        dl_2_relued = F.relu(dl_2, inplace=False)
        dl_3 = self.linear_128_1(dl_2_relued)
        scores = torch.reshape(dl_3, [features_1.shape[0], features_1.shape[1], 1])
        return scores

class CC_Rater(nn.Module):
    '''
    CC-RATER
    Combines the DNSMOS and the Classifier blocks to produce CCR scores out of two audios, which are divided in windows.
    Input: DNSMOS block input.
    Output: Classifier block output.
    '''
    def __init__(self):
        super(CC_Rater, self).__init__()
        self.dnsmos = DNSMOS()
        self.param_freeze(self.dnsmos)
        self.ccr_model = classifier()
        #for param in self.ccr_model.parameters():
        #    param.requires_grad = True
    def param_freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
    def forward(self, tensor_1, tensor_2):
        features_1 = self.dnsmos.forward(tensor_1)
        features_2 = self.dnsmos.forward(tensor_2)
        results = self.ccr_model(features_1, features_2)
        #result = results.mean(dim=1).requires_grad_(requires_grad=True)
        result = results.mean(dim=1)
        return result

#CODE INFRASTRUCTURE USED TO TRAIN AND TEST CC-RATER. COMPOSED BY:
# DATASET CLASS
# DATALOADER CLASS
# TRAINING LOOP
# TEST LOOP
# MAIN FUNCTION, WHICH IS DIFFERENT DEPENDING ON WHETHER YOU ARE TRAINING OR TESTING THE MODEL

class se_audio_dataset(torch.utils.data.Dataset):
    '''
    DATASET CLASS:
    Input:
    - a .csv file which contains the path to Audio A, the path to Audio B, and the CCR score of A wrt B.
    - the name of the root directory
    - the names of the three required .csv columns.
    Output: An item from this class contains:
    - The windows of Audio A
    - The windows of Audio B.
    -The CCR score of audio A wrt audio B.
    '''
    INPUT_LENGTH = 9.01
    FS = 16000
    def __init__(self, csv, rootdir, a_name=str, b_name=str, ccr_name=str):
        self.data = pd.read_csv(csv)
        self.rootdir = rootdir
        self.a_name = a_name
        self.b_name = b_name
        self.ccr_name = ccr_name
        return
    def __len__(self):
        return len(self.data)
    def audio_loader(self, fpath, input_length=INPUT_LENGTH, fs=FS):
        len_samples = int(input_length * fs)
        # Get the audios, downsample if needed
        if input_fs != fs:
            audio = librosa.resample(aud, input_fs, fs)
        else:
            audio = aud
        # Trim first second from audio, since many times it contains silence
        audio = audio[fs:]
        # Loop audios if they don't reach required length, if they are 0 length, set them to 144160 frames of silence
        # and go on. This allows you to spot empty audios and remove them from data.
        if len(audio) == 0:
            print(f"Audio {fpath} is empty. Replaced it with silence")
            audio = np.zeros(144160)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        return audio

    def get_windows(self, audio, num_hops, hop_len_samples, input_length=9.01, desired_windows=5):
        # The original DNSMOS code divided every audio into a number of overlapping windows proportional to its length:
        # [2, 144160], [4, 144160], [5, 144160], [7, 144160], [12, 144160], etc. This made it impossible for the data to be batched.
        # This function gets around this issue with two methods, depending whether the Original Number of Windows (ONW,
        # the Number of Windows originally produced by the DNSMOS code) is greater or lesser than the Desired Number of Windows (DNW).
        # For the following examples, DNW=5.
        # METHOD 1 - Audios for which ONW<DNW.
        # Audio is looped until DNW is reached. Example: audio of ONW [2, 144160]. First, it's duplicated -> [4, 144160].
        # When duplicating it would mean surpassing the DNW, then the slice [:DNW%ONW, :] is added -> [5, 144160]
        # METHOD 2 - Audios for which ONW > DNW
        # Originally, regardless of the audio length, DNSMOS always used the same hop_length (the distance between overlapping windows),
        # so num_hops (the number of points in the audio where overlapping windows start) varied depending on how long the audio was.
        # In order to always have the same num_hops and, therefore, the same number of windows, hop_length has to vary depending
        # on the length of the audio. The __getitem__ function takes care of this: when it sees that an audio's ONW is greater than the DNW,
        # it makes a custom hop_length for it that allows for it to be divided into the desired number of windows. This hop_length is
        # computed by performing the following operation:
        # HOP LENGTH = (AUDIO LENGTH - WINDOW SIZE) / (DESIRED NUMBER OF WINDOWS - 1)
        # EDIT: could've used padding to get same-size inputs
        window_list = []
        if num_hops >= desired_windows:
            for idx in range(desired_windows):
                slice_start = int(idx * hop_len_samples)
                window = torch.FloatTensor(audio[slice_start: slice_start + 144160])
                while window.shape < torch.Size([144160]):
                    window = torch.cat((window, torch.unsqueeze(window[-1], 0)), 0)
                window_list.append(window)
            windows = torch.cat([i.reshape(1, -1) for i in window_list])
        else:
            for idx in range(num_hops):
                window = torch.FloatTensor(audio[int(idx * hop_len_samples): int((idx + input_length) * hop_len_samples)])
                while window.shape != torch.Size([144160]):
                    window = torch.cat((window, torch.unsqueeze(window[-1], 0)), 0)
                window_list.append(window)
            windows = torch.cat([i.reshape(1, -1) for i in window_list])
            remainder = desired_windows % windows.size(dim=0)
            final_append = windows[:remainder, :]
            original_size = windows.size(dim=0)
            original_windows = windows[:, :]
            while windows.size(dim=0) < desired_windows:
                if original_size == 1:
                    windows = torch.cat((windows, original_windows), 0)
                else:
                    if (desired_windows/((windows.size(dim=0))+original_size)) < 1.00:
                        windows = torch.cat((windows, final_append), 0)
                    else:
                        windows = torch.cat((windows, original_windows), 0)
        windows.requires_grad_(requires_grad=True)
        return windows
    def __getitem__(self, idx):
        mos_ccr = self.data.iloc[idx][self.ccr_name]
        audio_a = self.audio_loader(self.data.iloc[idx][self.a_name])
        audio_b = self.audio_loader(self.data.iloc[idx][self.b_name])
        num_hops_a = int(np.floor(len(audio_a) / FS) - INPUT_LENGTH) + 1
        num_hops_b = int(np.floor(len(audio_b) / FS) - INPUT_LENGTH) + 1
        desired_windows = 5
        if num_hops_a >= desired_windows:
            hop_len_samples_a = int((len(audio_a) - 144160))/(desired_windows - 1)
        else:
            hop_len_samples_a = FS
        if num_hops_b >= desired_windows:
            hop_len_samples_b = int((len(audio_b) - 144160))/(desired_windows - 1)
        else:
            hop_len_samples_b = FS
        windows_a = self.get_windows(audio_a, num_hops=num_hops_a,hop_len_samples=hop_len_samples_a, input_length=INPUT_LENGTH, desired_windows=desired_windows)
        windows_b = self.get_windows(audio_b, num_hops=num_hops_b, hop_len_samples=hop_len_samples_b, input_length=INPUT_LENGTH, desired_windows=desired_windows)
        return windows_a, windows_b, mos_ccr

def train_loop(dataloader, model, loss_fn, optimizer, save_path):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("train")
    model = model.to(device)
    size = len(dataloader.dataset)
    for batch, (windows_a, windows_b, label) in enumerate(dataloader):
        windows_a = windows_a.cuda(device=device)
        windows_b = windows_b.cuda(device=device)
        label = (label.to(torch.float32)).cuda(device=device)
        #Compute prediction and loss
        prediction = model.forward(windows_a, windows_b)
        label = torch.unsqueeze(label, dim=1)
        loss = loss_fn(prediction, label)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(windows_a)
            print(f"loss: {loss},  [{current}/{size}]")
    torch.save(model.state_dict(), save_path)

def test_loop(dataloader, model, loss_fn, dataframe, output_csv):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("test")
    model = model.to(device)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    file_index = 1
    with torch.no_grad():
        for windows_a, windows_b, label in dataloader:
            windows_a = windows_a.cuda(device=device)
            windows_b = windows_b.cuda(device=device)
            label = (label.to(torch.float32)).cuda(device=device)
            label = torch.unsqueeze(label, dim=1)
            prediction = model.forward(windows_a, windows_b)
            loss = loss_fn(prediction, label).item()
            print(label.item(), prediction.item())
            new_row = {"File" : file_index, "Label": label.item(), "Prediction": prediction.item(), "Loss": loss}
            print(f"File number {file_index} was predicted to have a score of {prediction.data[0][0]}, its label is {label.data[0][0]}. Loss: {loss}")
            new_row_df = pd.DataFrame([new_row])
            dataframe = pd.concat([dataframe, new_row_df], axis=0, ignore_index=True)
            test_loss += loss_fn(prediction, label).item()
            correct += (prediction.argmax(1) == label).type(torch.float).sum().item()
            file_index += 1
    test_loss /= num_batches
    dataframe.index = np.arange(1, len(dataframe.index) + 1)
    dataframe.to_csv(output_csv)
    return test_loss
def main():
    # THE FOLLOWING BLOCK WAS USED TO TEST THE MODEL
    #######################################################################################
    # ROOT_DIR = "/home/your_dir/"
    # TEST_CSV = "/home/your_dir/testset_2.csv"
    # test_set = se_audio_dataset(TEST_CSV, ROOT_DIR, "A", "B", "CCR_MOS")
    # test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    # model = CC_Rater()
    # model.load_state_dict(torch.load("/home/your_dir/CC_Rater_69.pt"))
    # loss_function = nn.MSELoss()
    # column_names = ["File", "Label", "Prediction", "Loss"]
    # results_dataframe = pd.DataFrame(columns = column_names)
    # output_csv_path = "/home/your_dir/final_results.csv"
    # loss = test_loop(test_dataloader, model, loss_function, results_dataframe, output_csv_path)
    # print(f"CC-Rater performance on this test set is {loss}")
    ########################################################################################

    # THE FOLLOWING BLOCK WAS USED TO TRAIN THE MODEL
    ########################################################################################
    # session_log = []
    # ROOT_DIR = "/home/your_dir/"
    # TRAIN_CSV = "/home/your_dir/training_set.csv"
    # DEV_CSV = "/home/your_dir/dev_set.csv"
    # training_set = se_audio_dataset(TRAIN_CSV, ROOT_DIR, "A", "B", "CCR_MOS")
    # dev_set = se_audio_dataset(DEV_CSV, ROOT_DIR, "A", "B", "CCR_MOS")
    # training_dataloader = torch.utils.data.DataLoader(training_set, batch_size=8, shuffle=False)
    # dev_dataloader = torch.utils.data.DataLoader(dev_set, batch_size=8, shuffle=False)
    # SAVE_PATH = os.path.join(ROOT_DIR, "CC_Rater.pt")
    # model = CC_Rater()
    # loss_function = nn.MSELoss()
    # LEARNING_RATE = 0.001
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # EPOCHS = 200
    # epoch = 1
    # # Initialize it with a high loss
    # current_best = 5
    # while epoch <= EPOCHS:
    #     SAVE_PATH = os.path.join(ROOT_DIR, f"CC_Rater_{epoch}.pt")
    #     print(f"EPOCH {epoch}/{EPOCHS}")
    #     train_loop(training_dataloader, model, loss_function, optimizer, SAVE_PATH)
    #     print(f"Saved model from epoch {epoch} to file {SAVE_PATH}")
    #     epoch += 1
    #     loss = test_loop(dev_dataloader, model, loss_function)
    #     if loss <= current_best:
    #        current_best = loss
    #     message = f"EPOCH {epoch} of {EPOCHS}. Current MSE loss on dev set is {loss}. Current best loss is {current_best}"
    #     print(message)
    #     session_log.append(message)
    #     session_log_file = f"/home/your_dir/session_log_{epoch}.txt"
    #     with open(session_log_file, 'w') as file:
    #        file.write('\n'.join(session_log))
    ########################################################################################
if __name__ == "__main__":
    main()