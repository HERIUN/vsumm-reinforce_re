from __future__ import print_function
import os
import os.path as osp
import argparse
import h5py
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from utils import Logger, read_json, write_json, save_checkpoint
from models import *
from rewards import compute_reward
from generate_dataset import Generate_Dataset
import vsum_tools

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options
parser.add_argument('-s', '--source', type=str, default='', help="path to your custom video file") 
parser.add_argument('-d', '--dataset', type=str, default='', help="path to h5 dataset(optional)")

# Model options
parser.add_argument('--input-dim', type=int, default=1024, help="input dimension (default: 1024)")
parser.add_argument('--hidden-dim', type=int, default=256, help="hidden unit dimension of DSN (default: 256)")
parser.add_argument('--num-layers', type=int, default=1, help="number of RNN layers (default: 1)")
parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")
parser.add_argument('--weights', type=str, default='log/summe-split0/model_epoch60.pth.tar', help="pretrained DSN model parameters")

# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
parser.add_argument('--evaluate', action='store_true', help="whether to do evaluation only")
parser.add_argument('--save-dir', type=str, default='log', help="path to save output (default: 'log/')")
# parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--verbose', action='store_true', help="whether to show detailed test results")
# parser.add_argument('--save-results', action='store_true', help="whether to save output results")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False

@torch.no_grad()
def main():
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")
    start_time = time.time()
    if args.source != '' and args.dataset != None:
        print(f"Making dataset from your video {args.source}")
        gen_data = Generate_Dataset(args.source, args.dataset)
        gen_data.generate()
        gen_data.h5_file.close()
        print(f"dataset Done. {args.dataset}")

    print("Initialize dataset {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'a')
    num_videos = len(dataset.keys())
    print(f"# total videos : {num_videos}.")

    print(f"Initialize model")
    model = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    
    print(f"Load pretrained model : {args.weights}")
    model.load_state_dict(torch.load(args.weights))
    print(f"Load pretrained model Done.")
    
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    # start_time = time.time()
    model.eval()
    
    for id in range(num_videos):
        key = f'video_{id}'
        seq = dataset[key]['features'][...] # sequence of features, (seq_len, dim)
        seq = torch.from_numpy(seq).unsqueeze(0) # input shape (1, seq_len, dim)
        if use_gpu: seq = seq.cuda()
        probs = model(seq) # output shape (1, seq_len, 1)
        probs = probs.data.cpu().squeeze().numpy()
        cps = dataset[key]['change_points'][...]
        num_frames = dataset[key]['n_frames'][()]
        nfps = dataset[key]['n_frame_per_seg'][...].tolist()
        positions = dataset[key]['picks'][...]
        
        machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
        dataset.create_dataset(key + '/score', data=probs)
        dataset.create_dataset(key + '/machine_summary', data=machine_summary)
    
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    dataset.close()

if __name__ == '__main__':
    main()
