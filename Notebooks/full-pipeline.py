
# coding: utf-8

# All imports
import os, sys
import datetime
import time, shutil
import librosa
import librosa.display
import pysrt
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import subprocess

from utils import normalize, loss_cal, optim
from tensorflow.contrib import rnn
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage.filters import gaussian_filter
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.cluster import KMeans

if __name__ == "__main__":
    videoid = sys.argv[1]

    # # All configurations below:
    random_state = 222 # random seed
    # # K-Means offline clustering
    number_of_clusters = 2
    # Model param
    tisv_frame = 100 # max frame number of utterances of tdsv (lower values suffer) (100 means 1.025s)
    window = 0.025 # 25ms
    hop = 0.01 # 10ms This is frame level precision we will get
    # VAD param
    vad_threshold = 20 # threshold for voice activity detection
    # version number:
    version_num = f'-{tisv_frame}-{window}-{hop}-{vad_threshold}'
    audio_path = f'/datadrive2/dalon/diarization-experiments/Speaker_Verification/data/youtube-audio/{videoid}.wav'
    srt_path = f'/datadrive2/dalon/diarization-experiments/Speaker_Verification/data/srts/original/{videoid}.en.srt'
    embeddings_path = f'/datadrive2/dalon/diarization-experiments/Speaker_Verification/data/embedding/{videoid}-{version_num}.csv'
    save_srt_path = f'/datadrive2/dalon/diarization-experiments/Speaker_Verification/data/srts/outputs/{videoid}-{version_num}.en.srt'
    # model parameters
    model_path = '/datadrive2/dalon/diarization-experiments/Speaker_Verification/models'
    model_num = 9
    hidden = 768
    proj = 256
    num_layer = 3

    utter, sr = librosa.core.load(audio_path, sr=None)        # load audio
    utter_min_len = (tisv_frame * hop + window) * sr    # lower bound of utterance length
    # Get the duration
    duration = librosa.get_duration(utter, sr)
    # Duration of each window
    duration_per_frame = (duration / utter.shape[0])
    print(f'Duration: {duration}\nDuration per frame: {duration_per_frame}s\nMin length of utterance: {utter_min_len * duration_per_frame}s')
    tisv_frame_duration_s = utter_min_len * duration_per_frame
    intervals = librosa.effects.split(utter, top_db=vad_threshold)         # voice activity detection
    intervals_in_s = [[round(block[0] * duration_per_frame, 3), round(block[1] * duration_per_frame, 3)] for block in intervals]
    # pick the nfft atleast twice the size of window(whichs is the input) REF: https://stackoverflow.com/a/18080140/3959965
    # ft kernel size
    nfft = int(window // duration_per_frame) * 2

    new_intervals = []
    new_intervals_in_s = []
    # creating intervals list which are greater than the min utterance length
    # The last segment can still be less that utter_min_len, take care of it when you loop
    for idx, current_interval in enumerate(intervals):
        if (current_interval[1]-current_interval[0]) < utter_min_len:
            if not len(new_intervals):
                new_intervals.append([current_interval[0], current_interval[1]])
                new_intervals_in_s.append([intervals_in_s[idx][0], intervals_in_s[idx][1]])
            elif (new_intervals[-1][1] - new_intervals[-1][0]) >= utter_min_len:
                new_intervals.append([current_interval[0], current_interval[1]])
                new_intervals_in_s.append([intervals_in_s[idx][0], intervals_in_s[idx][1]])
            else:
                new_intervals[-1][1] = current_interval[1]
                new_intervals_in_s[-1][1] = intervals_in_s[idx][1]
        else:
            new_intervals.append([current_interval[0], current_interval[1]])
            new_intervals_in_s.append([intervals_in_s[idx][0], intervals_in_s[idx][1]])


    utterances_spec = []
    intervals_gt_s = []
    for idx, current_interval in enumerate(new_intervals):
        if (current_interval[1]-current_interval[0]) > utter_min_len:
            utter_part = utter[current_interval[0]:current_interval[1]]         # save first and last 180 frames of spectrogram.
            S = librosa.core.stft(y=utter_part, n_fft=nfft,
                                  win_length=int(window * sr), hop_length=int(hop * sr))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, n_mels=40)
            S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances

            prev_tisv_frame = 0
            prev_start = new_intervals_in_s[idx][0]
            for i in range(1, S.shape[1]//tisv_frame + 1):
                utterances_spec.append(S[:, prev_tisv_frame:tisv_frame * i])
                intervals_gt_s.append([prev_start, prev_start + tisv_frame_duration_s])
                prev_start = prev_start + tisv_frame_duration_s
                prev_tisv_frame = tisv_frame * i
            intervals_in_s[-1][1] = new_intervals_in_s[idx][1]  # Aligning the last interval

    utterances_spec = np.array(utterances_spec)
    utter_batch = np.transpose(utterances_spec, axes=(2,0,1))     # transpose [frames, batch, n_mels]
    print(utter_batch.shape)

    tf.reset_default_graph()
    batch_size = utter_batch.shape[1]
    verif = tf.placeholder(shape=[None, batch_size, 40], dtype=tf.float32)  # verification batch (time x batch x n_mel)
    batch = tf.concat([verif,], axis=1)

    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=hidden, num_proj=proj) for i in range(num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize

    print("embedded size: ", embedded.shape)

    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("model path :", model_path)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_path)
        ckpt_list = ckpt.all_model_checkpoint_paths
        loaded = 0
        for model in ckpt_list:
            if model_num == int(model[-1]):    # find ckpt file which matches configuration model number
                print("ckpt file is loaded !", model)
                loaded = 1
                saver.restore(sess, model)  # restore variables from selected ckpt file
                break
        if loaded == 0:
            raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")
        data = sess.run(embedded, feed_dict={verif:utter_batch})

    # cossine similarity
    similarity = np.dot(data, data.T)
    # squared magnitude of preference vectors (number of occurrences) (diagonals are ai*ai)
    square_mag = np.diag(similarity)
    # inverse squared magnitude
    inv_square_mag = 1 / square_mag
    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)
    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    A =  cosine.T * inv_mag
    # Fill the diagonals with very large negative value
    np.fill_diagonal(A, -1000)
    # Fill the diagonals with the max of each row
    np.fill_diagonal(A, A.max(axis=1))
    # final step in cossine sim
    A = (1-A)/2
    # Gaussian blur
    sigma = 0.5 # we will select sigma as 0.5
    A_gau = gaussian_filter(A, sigma)
    # Thresholding using multiplier = 0.01
    threshold_multiplier = 0.01
    A_thresh = A_gau * threshold_multiplier
    # Symmetrization
    A_sym = np.maximum(A_thresh, A_thresh.T)
    # Diffusion
    A_diffusion = A_sym * A_sym.T
    # Row-wise matrix Normalization
    Row_max = A_diffusion.max(axis=1).reshape(1, A_diffusion.shape[0])
    A_norm = A_diffusion / Row_max.T
    # Eigen decomposition
    eigval, eigvec = np.linalg.eig(A_norm)
    # Since eigen values cannot be negative for Positive semi definite matrix, the numpy returns negative values, converting it to positive
    eigval = np.abs(eigval)
    # reordering eigen values
    sorted_eigval_idx = np.argsort(eigval)[::-1]
    sorted_eigval = np.sort(eigval)[::-1]
    # For division according to the equation
    eigval_shifted = np.roll(sorted_eigval, -1)
    # Thresholding eigen values because we don't need very low eigan values due to errors
    eigval_thresh = 0.1
    sorted_eigval = sorted_eigval[sorted_eigval > eigval_thresh]
    eigval_shifted = eigval_shifted[:sorted_eigval.shape[0]]
    # Don't take the first value for calculations, if first value is large, following equation will return k=1, and we want more than one clusters
    # Get the argmax of the division, since its 0 indexed, add 1
    k = np.argmax(sorted_eigval[1:]/eigval_shifted[1:]) + 2
    print(f'Number of Eigen vectors to pick: {k}')
    # Get the indexes of eigen vectors
    idexes = sorted_eigval_idx[:k]
    A_eigvec = eigvec[:, idexes]
    np.savetxt(embeddings_path, A_eigvec, delimiter='\t') # embeddings for viz

    A_eigvec_norm = sk_normalize(A_eigvec) # l2 normalized
    kmeans = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=random_state)
    kmeans.fit(A_eigvec)
    labels = kmeans.labels_

    subs = pysrt.open(srt_path, encoding="utf-8")
    convert_to_s = lambda st: (st.hours * 60 * 60) + (st.minutes * 60) + (st.seconds) #+ \
                                #(st.milliseconds / 1000)
    get_start_and_end = lambda sub: (convert_to_s(sub.start), convert_to_s(sub.end))

    for sub in subs:
        start, end = get_start_and_end(sub)
        speakers = []
    #     speakers_intervals = []
        for idx, interval in enumerate(intervals_gt_s):
            interval[0], interval[1] = int(interval[0]), int(interval[1])
            if interval[0] <= start <= interval[1] or interval[0] <= end <= interval[1]            or (start <= interval[0] and interval[1] <= end):
                speakers.append(labels[idx])
                
        if speakers:
    #         print(speakers)
            sp_list, sp_count = np.unique(speakers, return_counts=True)
            speaker_dist = 'Speakers: '
            number_speakers = len(speakers)
            for idx, sp in enumerate(sp_list):
                speaker_dist += f'{sp}, '
            sub.text = f'{speaker_dist[:-2]} S:{speakers} T:{end - start} {sub.text}'
    subs.save(save_srt_path, encoding='utf-8')
    print(f'Completed! Find the srt in {save_srt_path}')
