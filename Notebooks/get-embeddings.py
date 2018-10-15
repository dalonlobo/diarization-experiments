import tensorflow as tf
import numpy as np
import os
import time
from utils import random_batch, normalize, similarity, loss_cal, optim
from configuration import get_config
from tensorflow.contrib import rnn

config = get_config()

path = config.model_path
tf.reset_default_graph()

# draw graph
# enroll is ground truth 
# verif is the actual output
enroll = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32) # enrollment batch (time x batch x n_mel)
verif = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32)  # verification batch (time x batch x n_mel)
batch = tf.concat([enroll, verif], axis=1)

# embedding lstm (3-layer default)
with tf.variable_scope("lstm"):
    lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
    lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
    outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
    embedded = outputs[-1]                            # the last ouput is the embedded d-vector
    embedded = normalize(embedded)                    # normalize

print("embedded size: ", embedded.shape)

# enrollment embedded vectors (speaker model)
enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:config.N*config.M, :], shape= [config.N, config.M, -1]), axis=1))
# verification embedded vectors
verif_embed = embedded[config.N*config.M:, :]

similarity_matrix = similarity(embedded=verif_embed, w=1., b=0., center=enroll_embed)

saver = tf.train.Saver(var_list=tf.global_variables())
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # load model
    print("model path :", path)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(path, "Check_Point"))
    ckpt_list = ckpt.all_model_checkpoint_paths
    loaded = 0
    for model in ckpt_list:
        if config.model_num == int(model[-1]):    # find ckpt file which matches configuration model number
            print("ckpt file is loaded !", model)
            loaded = 1
            saver.restore(sess, model)  # restore variables from selected ckpt file
            break

    if loaded == 0:
        raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

    print("test file path : ", config.test_path)

    # return similarity matrix after enrollment and verification
    time1 = time.time() # for check inference time
    if config.tdsv:
        S = sess.run(similarity_matrix, feed_dict={enroll:random_batch(shuffle=False, noise_filenum=1),
                                                   verif:random_batch(shuffle=False, noise_filenum=2)})
    else:
        S = sess.run(verif_embed, feed_dict={enroll:random_batch(shuffle=False),
                                                   verif:random_batch(shuffle=False, utter_start=config.M)})
        print(type(S))
        print(S.shape)
        print(S)

data_path = f'viz/data-{config.N}-{config.M}.tsv'
labels_path = f'viz/labels-{config.N}-{config.M}.tsv'

with open(labels_path, 'w') as l:
    np.savetxt(data_path, S, delimiter="\t")
    speaker = 1
    for i in range(1, config.N * config.M + 1):
        l.write(f'speaker-{speaker}\n')
        if i % config.M == 0:
            speaker += 1
    