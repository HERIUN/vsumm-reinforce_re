import h5py
import os
import cv2
import numpy as np
import math
from PIL import Image
from tqdm import tqdm
import argparse

import torch
from torchvision.models import googlenet, resnet152
from torchvision import transforms
from KTS import cpd_auto

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options

"""
convert video files (.json) to h5 file format
"""


parser.add_argument('--input', '--split', type=str, required=True, help="input videos path or video file ex. ./video_folder/ or ./video.mp4")
parser.add_argument('--output', type=str, default=None, help="output h5 data ex")


class Generate_Dataset:
    def __init__(self, video_path, save_h5_path):
        self.device = torch.device('cuda:0')
        self.googlenet = googlenet(pretrained=True)
        self.extractor = torch.nn.Sequential(*list(self.googlenet.children())[:-2]).to(self.device)
        self.video_list = []
        self.dataset = {}
        self.h5_file = h5py.File(save_h5_path, 'w')
        self._set_video_list(video_path)
        
        self.preprocess = transforms.Compose([ # https://pytorch.org/hub/pytorch_vision_googlenet/
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    def _set_video_list(self, video_path):
        if os.path.isdir(video_path):
            self.video_path = video_path
            self.video_list = os.listdir(video_path)
        else:
            self.video_path = ''
            self.video_list.append(video_path)
        
        for idx, file_name in enumerate(self.video_list):
            self.dataset[f'video_{idx}'] = {}
            self.h5_file.create_group(f'video_{idx}')
    
    def _extract_feature(self, frame): # frame's shape=(H,W,C)
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
        im = Image.fromarray(im) # cv2 to PIL
        im = self.preprocess(im)
        im = im.unsqueeze(0).to(self.device) # it should be shape : (1,3,224,224)
        
        with torch.no_grad():
            feature = self.extractor(im).cpu().numpy().flatten() # [1(N), 1024, 1, 1] -> [1024]
    
        return feature
       
    def _get_change_points(self, feature_stack_for_cps, n_frames, fps):
        n = n_frames / fps
        m = int(math.ceil(n/2.0)) # maximum number of change points
        K = np.dot(feature_stack_for_cps, feature_stack_for_cps.T)
        cps, _ = cpd_auto(K, m, 1) # change points
        cps = np.concatenate(([0], cps, [n_frames-1]))

        temp_change_points = []
        for idx in range(len(cps)-1):
            segment = [cps[idx], cps[idx+1]-1]
            if idx == len(cps)-2:
                segment = [cps[idx], cps[idx+1]]

            temp_change_points.append(segment)
        cps = np.array(list(temp_change_points))

        temp_n_frame_per_seg = []
        for cps_idx in range(len(cps)):
            n_frame = cps[cps_idx][1] - cps[cps_idx][0] + 1 # include self frame
            temp_n_frame_per_seg.append(n_frame)
        n_frame_per_seg = np.array(list(temp_n_frame_per_seg))

        return cps, n_frame_per_seg
    
    def generate(self):
        for idx, file_name in enumerate(self.video_list):
            video_path = file_name
            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, file_name)
            base_name = os.path.basename(video_path).split('.')[0]
            if not os.path.exists(os.path.join('./', base_name)):
                os.mkdir(os.path.join('./', base_name))
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) # for fps // 2 frame sampling
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            picks = []
            feature_stack_for_cps = None 
            feature_stack_for_train = None
            
            print("extracting features")
            for frame_idx in tqdm(range(n_frames-1)):
                success, frame = cap.read()
                if success:
                    frame_feat = self._extract_feature(frame)
                    
                    if frame_idx % 15 == 0:
                        picks.append(frame_idx)
                        if feature_stack_for_train is None:
                            feature_stack_for_train = frame_feat
                        else:
                            feature_stack_for_train = np.vstack((feature_stack_for_train, frame_feat))
                
                    if feature_stack_for_cps is None:
                        feature_stack_for_cps = frame_feat
                    else:
                        feature_stack_for_cps = np.vstack((feature_stack_for_cps, frame_feat))
                    
                else:
                    break    
                # frame_name = f'{frame_idx}.jpg'
                # cv2.imwrite(os.path.join('./', base_name, frame_name), frame)
                
            cap.release()
            
            print("getting auto change points by KTS")
            cps, n_frame_per_seg = self._get_change_points(feature_stack_for_cps, n_frames, fps)
            
            self.h5_file['video_{}'.format(idx)]['features'] = list(feature_stack_for_train)
            self.h5_file['video_{}'.format(idx)]['picks'] = np.array(list(picks))
            self.h5_file['video_{}'.format(idx)]['n_frames'] = n_frames
            self.h5_file['video_{}'.format(idx)]['fps'] = fps
            self.h5_file['video_{}'.format(idx)]['change_points'] = cps
            self.h5_file['video_{}'.format(idx)]['n_frame_per_seg'] = n_frame_per_seg

if __name__ == '__main__':
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input.split(os.path.sep)[-1]+'.h5'
    gen = Generate_Dataset(args.input, args.output)
    gen.generate()
    gen.h5_file.close()
    
                    