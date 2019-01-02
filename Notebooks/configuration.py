import argparse
import numpy as np

parser = argparse.ArgumentParser()    # make parser


# get arguments
def get_config():
    config, unparsed = parser.parse_known_args()
    return config


# return bool type of argument
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Data Preprocess Arguments
data_arg = parser.add_argument_group('Data')
data_arg.add_argument('--noise_path', type=str, default='/datadrive2/dalon/diarization-experiments/Speaker_Verification/data/VCTK-Corpus/noise', help="noise dataset directory")
data_arg.add_argument('--train_path', type=str, default='/datadrive2/dalon/diarization-experiments/Speaker_Verification/data/VCTK-Corpus/train', help="train dataset directory")
data_arg.add_argument('--test_path', type=str, default='/datadrive2/dalon/diarization-experiments/Speaker_Verification/data/VCTK-Corpus/test', help="test dataset directory")
data_arg.add_argument('--tdsv', type=str2bool, default=False, help="text dependent or not")
data_arg.add_argument('--sr', type=int, default=16000, help="sampling rate")
data_arg.add_argument('--nfft', type=int, default=512, help="fft kernel size")
data_arg.add_argument('--window', type=float, default=0.025, help="window length") # 25ms
data_arg.add_argument('--hop', type=float, default=0.01, help="hop size") # 10ms
data_arg.add_argument('--tdsv_frame', type=int, default=160, help="frame number of utterance of tdsv")
data_arg.add_argument('--tisv_frame', type=int, default=160, help="max frame number of utterances of tdsv")
data_arg.add_argument('--tisv_frame_min', type=int, default=50, help="min frame length to confirm with 'd-vector v-3 model'")

# Model Parameters
model_arg = parser.add_argument_group('Model')
model_arg.add_argument('--hidden', type=int, default=768, help="hidden state dimension of lstm")
model_arg.add_argument('--proj', type=int, default=256, help="projection dimension of lstm")
model_arg.add_argument('--num_layer', type=int, default=3, help="number of lstm layers")
model_arg.add_argument('--restore', type=str2bool, default=False, help="restore model or not")
model_arg.add_argument('--model_path', type=str, default='./tisv_model', help="model directory to save or load")
model_arg.add_argument('--model_num', type=int, default=3, help="number of ckpt file to load")

# Training Parameters
train_arg = parser.add_argument_group('Training')
train_arg.add_argument('--train', type=str2bool, default=False, help="train session or not(test session)")
train_arg.add_argument('--N', type=int, default=4, help="number of speakers of batch")
train_arg.add_argument('--M', type=int, default=5, help="number of utterances per speaker")
train_arg.add_argument('--noise_filenum', type=int, default=16, help="how many noise files will you use")
train_arg.add_argument('--loss', type=str, default='softmax', help="loss type (softmax or contrast)")
train_arg.add_argument('--optim', type=str.lower, default='sgd', help="optimizer type")
train_arg.add_argument('--lr', type=float, default=1e-3, help="learning rate")
train_arg.add_argument('--beta1', type=float, default=0.5, help="beta1")
train_arg.add_argument('--beta2', type=float, default=0.9, help="beta2")
train_arg.add_argument('--iteration', type=int, default=100000, help="max iteration")
train_arg.add_argument('--comment', type=str, default='', help="any comment")
train_arg.add_argument('--max_batch_utterances', type=int, default=1000, help="number of utterances of one speaker")

config = get_config()
print(config)           # print all the arguments
