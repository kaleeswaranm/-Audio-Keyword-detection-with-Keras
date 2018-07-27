Keyword detection using stacked LSTM model with a time distributed dense layer between them.

The model takes in the audio clips and converts it into MFCCs (MFCC - https://en.wikipedia.org/wiki/Mel-frequency_cepstrum).
These MFCCs are passed into the stacked LSTM model and final softmax layer predicts if the audio clip pronounces one of the
four classes of ['yes', 'no', 'happy', 'stop'].

The model produces around 94% accuracy on validation data at the end of 50 epochs.
