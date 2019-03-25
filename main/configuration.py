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

# Arguments
data_arg = parser.add_argument_group('Data')
data_arg.add_argument('--random_state', type=int, default=123)
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
model_arg.add_argument('--model_num', type=int, default=3, help="number of ckpt file to load")

# Main program parameters
main_arg = parser.add_argument_group('Main')
main_arg.add_argument('--number_of_speakers', type=int, default=2, help="Number of speakers")
main_arg.add_argument('--log_path', type=str, default='main.logs')
main_arg.add_argument('--output_dir', required=True, type=str, default='output.json', help="Path to save the output json file")
main_arg.add_argument('--audio_file', required=True, type=str, default='xxx.wav', help="Enter the absolute path to audio file")
main_arg.add_argument('--srt_path', required=True, type=str, default='xxx.en.srt', help="Enter the absolute path to srt file")
main_arg.add_argument('--model_path', type=str, default='models/model.ckpt-46', help="model directory to save or load")


config = get_config()
print(config)           # print all the arguments
