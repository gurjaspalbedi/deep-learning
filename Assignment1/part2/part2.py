# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 14:35:53 2019

@author: gurjaspal
"""
#%%
import torch.nn as nn
import librosa
import numpy as np
import torch

#%%
s, sr=librosa.load('data/train_clean_male.wav', sr=None)
S=librosa.stft(s, n_fft=1024, hop_length=512)
sn, sr=librosa.load('data/train_dirty_male.wav', sr=None)
X=librosa.stft(sn, n_fft=1024, hop_length=512)

#%%
s_abs = np.abs(S)
x_abs = np.abs(X)
#%%
s = torch.tensor(np.transpose(s_abs)).cuda()
x = torch.tensor(np.transpose(x_abs)).cuda()
BATCH = 171

#%%
train_loader = torch.utils.data.DataLoader(x, batch_size=BATCH)
test_loader = torch.utils.data.DataLoader(s, batch_size=BATCH)

#%%
class NeuralNet(nn.Module):
  def __init__(self):
    super(NeuralNet, self).__init__()
    self.input_layer = nn.Linear(513, 2459)
    self.hidden_layer1 = nn.Linear(2459, 2459)
    self.hidden_layer2 = nn.Linear(2459, 2459)
    self.output_layer = nn.Linear(2459, 513)
    
    self.activation = nn.ReLU()

  def forward(self, data):
    print(data.shape)
    data = data.view(-1, 513).cuda()
    output = self.activation(self.input_layer(data))
    output = self.activation(self.hidden_layer1(output))
    output = self.activation(self.hidden_layer2(output))
    output_layer = self.output_layer(output)
    final = torch.nn.functional.relu(output_layer)
    return final


#%%
neural_network = NeuralNet()
neural_network = neural_network.cuda()
#network_output , torch.max(target_set, 1)[1]
# def loss_function(ypred,yactual):
#   return torch.sum((ypred - yactual) ** 2)
loss_function= nn.MSELoss()
para = neural_network.parameters()
optimizer = torch.optim.Adam(params=para, lr=0.01)

#%%
def get_hadamard_product_result(x_test, x_test_abs, s_hat_test):
    div = x_test / x_test_abs
    return torch.tensor(div) * s_hat_test

#%%
input_iter = iter(train_loader)
target_iter = iter(test_loader)

for i in range(s.shape[0]//BATCH):
    print(i)
    input_set = input_iter.next()
    target_set = target_iter.next()
    
    input_set = input_set.cuda()
    target_set = target_set.cuda()
    optimizer.zero_grad()
    network_output  = neural_network(input_set)
    loss = loss_function(network_output , target_set)
    loss.backward()
    optimizer.step()
    
#%%
st, sr=librosa.load('data/test_x_01.wav', sr=None)
test_01 = librosa.stft(st, n_fft=1024, hop_length=512)
test_01_abs = torch.tensor(np.abs(test_01))
test_01_abs = np.transpose(test_01_abs)
# test_01 = torch.tensor(test_01)
t_loader = torch.utils.data.DataLoader(test_01_abs, batch_size=142)
#%%  
with torch.no_grad():
    t_iter = iter(t_loader)
    for i in range(test_01_abs.shape[0]//142):
        data = t_iter.next()
        data = data.cuda()
        output = neural_network(data)
        print(output.shape)


#%%
# speech_spectogram = get_hadamard_product_result(test_01, np.abs(test_01), output.numpy())
spec = (test_01 / np.abs(test_01)) * output.cpu().numpy().T
#%%

#%%

#%%
spec_istft = librosa.istft(spec, length=512, win_length=1024)


#%%
librosa.output.write_wav('test_s_01_recons.wav', spec_istft, sr)

#%%
librosa.output.write_wav('test_s_01_recons.wav', spec_istft, sr)