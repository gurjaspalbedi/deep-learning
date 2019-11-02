# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:01:58 2019

@author: gurjaspal
"""

import torch.nn as nn
import librosa
import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import IPython.display as ipd
import os

print(os.getcwd())
x, srx=librosa.load('homework3/timit-homework/tr/trn0000.wav', sr=None)
X=librosa.stft(x, n_fft=1024, hop_length=512)