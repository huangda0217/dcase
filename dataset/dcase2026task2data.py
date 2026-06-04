from torch.utils.data import Dataset
import os
import librosa
import soundfile as sf
import re
import tqdm
import numpy as np
import torch
from inference.preprocess import wiener_denoise
import warnings
warnings.filterwarnings("ignore")

def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    info = sf.info(wav_name)
    ch = info.channels
    assert any([
        mono and ch == 1,
        not mono and ch > 1,
    ]), f"Channel mismatch: The file is {ch=}, but {mono=} is specified."

    if mono:
        return librosa.load(wav_name, sr=None, mono=mono)
    else:
        y, sr = librosa.load(wav_name, sr=None, mono=mono)
        return y, sr

class DCSAE2026Task2Data(Dataset):
    def __init__(self, data_dir,mode='train',mono=False): #data_dir=data/dcase2026t2/dev_data/raw
        self.data_dir = data_dir
        self.mode = mode
        self.machine_type,self.file_list = self.__get_file_list(data_dir,mode)
        self.data_list,self.label_list = self.__get_data(self.file_list,self.machine_type)


    def __get_file_list(self,data_dir,mode):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {os.path.abspath(data_dir)}. \nPlease download the dataset and place it in the correct path."
            )
        machine_type=[i for i in  os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, i))]
        file_list=[]
        for i in machine_type:
            basename=os.path.join(data_dir,i,mode)
            for root, _, files in os.walk(basename):
                for file in files:
                    if file.endswith(".wav"):
                        file_list.append(os.path.join(root,file))
        return machine_type,file_list
    
    def __get_data(self, file_list,machine_type):
        data_list=[]
        label_list=[]
        loop=tqdm.tqdm(file_list,desc=f'generating {self.mode} data for all machines...')
        for  i in loop:
            data, _ = file_load(i)
            if data.ndim > 1:
                ch1 = data[0, :]
                ch2 = data[1, :]
                data = wiener_denoise(ch1, ch2)

            data=torch.from_numpy(data)
            for index,type in enumerate(machine_type):
                if re.search(r'\b' + re.escape(type) + r'\b', i):
                    label_list.append(index)
            data_list.append(data)


        return data_list,label_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]
        file_name = self.file_list[index]
        return data, label,file_name

