# All imports
import os
import logging
import datetime
import json

import librosa
import pysrt
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn
from utils import normalize
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.cluster import KMeans
from scipy.ndimage.filters import gaussian_filter
from configuration import get_config

config = get_config()
log_file = os.path.abspath(config.log_path)
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
    )
print(f'Log path: {log_file}')

def main():
    # Changing to 25, which will give slightly better intervals, 20 gives very short intervals
    vad_threshold = 25 # threshold for voice activity detection

    # Data prep
    # I'm saving only 2 embeddings i.e. first and last tisv_frames for given interval in an audio. So each .npy
    # embedding file will have a shape of (2, 256)
    tf.reset_default_graph()
    batch_size = 2 # Fixing to 2 since we take 2 for each interval #utter_batch.shape[1]
    verif = tf.placeholder(shape=[None, batch_size, 40], dtype=tf.float32)  # verification batch (time x batch x n_mel)
    batch = tf.concat([verif,], axis=1)
    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize
    config_tensorflow = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
    saver = tf.train.Saver(var_list=tf.global_variables())
    # Extract embeddings
    # Each embedding saved file will have (2, 256)
    with tf.Session(config=config_tensorflow) as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, config.model_path)
        logging.info("loading audio")
        audio_path = config.audio_file
        utter, sr = librosa.core.load(audio_path, sr=config.sr)        # load audio
        utter_min_len = (config.tisv_frame_min * config.hop + config.window) * sr    # lower bound of utterance length
        # Get the duration
        duration = librosa.get_duration(utter, sr)
        # Duration of each window
        duration_per_frame = (duration / utter.shape[0])
        logging.info(f'Duration: {duration}\nDuration per frame: {duration_per_frame}s\nMin length of utterance: {utter_min_len * duration_per_frame}s')
        tisv_frame_duration_s = utter_min_len * duration_per_frame
        intervals = librosa.effects.split(utter, top_db=vad_threshold)         # voice activity detection

        all_data = []
        logging.info('Converting intervals to embeddings')
        selected_intervals_idx = []
        for idx, current_interval in enumerate(intervals):
            if (current_interval[1]-current_interval[0]) > utter_min_len:
                # Save these selected intervals, as shorter ones are ignored
                selected_intervals_idx.append(idx)
                utterances_spec = []
                utter_part = utter[current_interval[0]:current_interval[1]]         # save first and last 160 frames of spectrogram.
                S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
                                        win_length=int(config.window * sr), hop_length=int(config.hop * sr))
                S = np.abs(S) ** 2
                mel_basis = librosa.filters.mel(sr=sr, n_fft=config.nfft, n_mels=40)
                S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                utterances_spec.append(S[:, :config.tisv_frame])
                utterances_spec.append(S[:, -config.tisv_frame:])
                utterances_spec = np.array(utterances_spec)
                utter_batch = np.transpose(utterances_spec, axes=(2,0,1))     # transpose [frames, batch, n_mels]

                data = sess.run(embedded, feed_dict={verif:utter_batch})
                all_data.extend(data)
    data = np.array(all_data)

    # # Spectral clustering
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
    logging.debug(f'Number of Eigen vectors to pick: {k}')
    # Get the indexes of eigen vectors
    idexes = sorted_eigval_idx[:k]
    A_eigvec = eigvec[:, idexes]
    A_eigvec = A_eigvec.astype('float32')

    # # K-Means offline clustering
    A_eigvec_norm = sk_normalize(A_eigvec) # l2 normalized
    kmeans = KMeans(n_clusters=config.number_of_speakers, init='k-means++', random_state=config.random_state)
    kmeans.fit(A_eigvec)
    labels = kmeans.labels_
    output_srt_json = os.path.join(config.output_dir, os.path.basename(config.srt_path) + '.json')
    output_wav_json = os.path.join(config.output_dir, os.path.basename(config.srt_path) + '.wav.json')
    OL_INDICATOR = 'OL'
    SIL_INDICATOR = -1
    json_data = []
    for idx, i in enumerate(selected_intervals_idx):
        start = str(datetime.timedelta(seconds = intervals[i][0] * duration_per_frame))
        end = str(datetime.timedelta(seconds = intervals[i][1] * duration_per_frame))
        speaker = labels[idx*2]
        if labels[idx*2] != labels[(idx*2)+1]:
            speaker = 'OL' # possible overlap
        json_data.append({
            'start': start,
            'end': end,
            'speaker': str(speaker)
        })
    # Save the output to json
    with open(output_wav_json, 'w') as f:
        json.dump(json_data, f, indent=4)

    complete_json = {}
    json_data = []
    subs = pysrt.open(config.srt_path, encoding="utf-8")
    convert_to_ms = lambda st: (st.hours * 60 * 60 * 1000) + \
                                (st.minutes * 60 * 1000) +\
                                (st.seconds * 1000) +\
                                (st.milliseconds)
    for sub in subs:
        start_in_ms = convert_to_ms(sub.start)
        end_in_ms = convert_to_ms(sub.end)
        speakers = []
        for idx, i in enumerate(selected_intervals_idx):
            start = intervals[i][0] * duration_per_frame * 1000
            end = intervals[i][1] * duration_per_frame * 1000
            if start_in_ms <= start <= end_in_ms:
                speaker = int(labels[idx*2])
                if labels[idx*2] != labels[(idx*2)+1]:
                    speaker = OL_INDICATOR # possible overlap
                speakers.append(speaker)
        json_data.append({
            "index": sub.index,
            "start": sub.start.to_time().strftime("%H:%M:%S,%f")[:-3],
            "end": sub.end.to_time().strftime("%H:%M:%S,%f")[:-3],
            'speakers': np.unique(speakers).tolist(),
            'speakers_distribution': speakers,
            'text': sub.text
        })
    metadata = {
        "overlap_indicator": OL_INDICATOR,
        "duration": duration,
        "class_names": np.unique(labels).tolist(),
        "num_of_speakers": len(set(labels)),
        "silence_indicator": SIL_INDICATOR
    }
    complete_json["metadata"] = metadata
    complete_json["srt"] = json_data
    # Save the output to json
    with open(output_srt_json, 'w') as f:
        json.dump(complete_json, f, indent=4)

if __name__ == "__main__":
    """
    Speaker diarization program:
    input: audio file
    output: json with following format:
            {
                'start': start-time,
                'end': end-time,
                'speaker': speaker
            }
    """
    main()
    print('Program completed!')