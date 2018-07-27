import numpy as np
import os
import pickle
from scipy.io import wavfile
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import LSTM, BatchNormalization, GRU, Dropout, Dense, Activation, Conv2D, MaxPooling2D, Flatten, TimeDistributed, Bidirectional
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import optimizers
import librosa
import librosa.display
import gc
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

#function to get the mfcc

def wav2mfcc(wav_file):
    mfcc = librosa.feature.melspectrogram(wav_file.astype('float32'), sr=wav_file.shape[0], n_fft = 600, hop_length = 200)
    log_S = librosa.power_to_db(mfcc, ref=np.max)
    return(log_S)

#optional function if spectrogram will be fed to model

"""
def graph_spectrogram(wav_file):
    nfft = 200
    fs = wav_file.shape[0]
    noverlap = 120
    pxx, freqs, bins, im = plt.specgram(wav_file, nfft, fs, noverlap = noverlap)
    return pxx
"""

directory = '/home/kaleeswaran/Desktop/Speech Recognition'

#target classes that will be used

target = ['yes', 'no', 'happy', 'stop']

eg = '/home/kaleeswaran/Desktop/Speech Recognition/yes/0a7c2a8d_nohash_0.wav'
rate, egd = wavfile.read(eg)
egs = wav2mfcc(egd)

"""
#doing some plotting 
plt.subplot(211)
plt.plot(egd)
plt.subplot(212)
librosa.display.specshow(egs)
plt.show()
"""

x = []
y = []

mapping = {b: a for a, b in enumerate(target)}

#reading the audio files and converting to wave forms
#the original audio is sampled at 16000 sample rate and each clip is 1 sec length
#if a clip is of length less than 1 sec, it will be padded with 0 at the
#beginning and end to get length 16000 and mfcc will be calculated from here

for folder in target:
    print(folder)
    path = os.listdir(directory + '/' + folder)
    for sound in path:
        rate, clip = wavfile.read(directory + '/' + folder + '/' + sound)
        if clip.shape[0] != 16000:
            pad_width = 16000 - clip.shape[0]
            clip = np.pad(clip, (int(pad_width / 2), pad_width - int(pad_width / 2)), 'constant', constant_values = np.mean(clip))
        x.append(clip)
        y.append(mapping[folder])

"""
#doing some plotting
rte, cte = wavfile.read(directory + '/' + 'yes' + '/' + 'e98cb283_nohash_1.wav')

plt.subplot(221)
plt.plot(cte)
plt.subplot(222)
librosa.display.specshow(wav2mfcc(cte))
plt.subplot(223)
plt.plot(np.pad(cte, (int(pad_width / 2), pad_width - int(pad_width / 2)), 'constant', constant_values = np.mean(cte)))
plt.subplot(224)
librosa.display.specshow(wav2mfcc(np.pad(cte, (int(pad_width / 2), pad_width - int(pad_width / 2)), 'constant', constant_values = np.mean(cte))))
plt.show()
"""

#fetching the mfcc

specvec = np.zeros((len(x), 128, 81))

for i, clip in enumerate(x):
    print(i)
    specvec[i] = wav2mfcc(clip)

#shuffling

idxes = np.random.permutation(specvec.shape[0])
X = specvec[idxes]
y = np.array(y)
y = y[idxes]

#pickling

pickle.dump(X, open((directory + '/' + 'X.p'), 'wb'))
pickle.dump(y, open((directory + '/' + 'y.p'), 'wb'))

#loading

X = pickle.load(open((directory + '/' + 'X.p'), 'rb'))
y = pickle.load(open((directory + '/' + 'y.p'), 'rb'))

#reshaping the data to feed into our stacked LSTM model

Xd = X.transpose(0,2,1)

#90/10 train test split

X_train = Xd[:int(0.9*Xd.shape[0])]
X_test  = Xd[int(0.9*Xd.shape[0]):]
y_train = y[:int(0.9*Xd.shape[0])]
y_test  = y[int(0.9*Xd.shape[0]):]

#conversion to categorical

Y_train = to_categorical(y_train, num_classes = 4)
Y_test  = to_categorical(y_test, num_classes = 4)

#model architecture - a stacked LSTM model with two LSTMs stacked over each
#other with a dense layer between them
#using Adam optimizer to train with softmax activation

model = Sequential()
model.add(LSTM(64, return_sequences = True, input_shape = (81, 128)))
model.add(TimeDistributed(Dense(64)))
model.add(LSTM(32))
model.add(Dense(4, activation = 'softmax'))

opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=50, batch_size = 32, validation_data=(X_test, Y_test))

"""
#doing some plotting
i = 0
for key, values in history.history.items():
    plt.subplot(2,2,i+1)
    plt.plot(history.history[key])
    plt.title(key)
    i+=1
"""

#getting the confusion matrix

Y = model.predict_classes(X_test)
cm = confusion_matrix(y_test, Y)
df_cm = pd.DataFrame(cm, index = target, columns = target)
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True)