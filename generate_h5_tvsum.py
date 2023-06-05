import h5py
import os
import cv2
import numpy as np
import math
from PIL import Image
from tqdm import tqdm
import argparse
import re

import torch
from torchvision.models import googlenet, resnet152
from torchvision import transforms
from KTS import cpd_auto

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options

parser.add_argument('--origin', type=str, required=True, help="orgin summe h5 path ")
parser.add_argument('--input', type=str, required=True, help="input tvsum video path ex. ../new_database/")
parser.add_argument('--output', type=str, default=None, help="output h5 data ex")


class Generate_Dataset:
    def __init__(self, origin_h5, video_path, save_h5_path):
        self.device = torch.device('cuda:0')
        self.googlenet = googlenet(pretrained=True)
        self.extractor = torch.nn.Sequential(*list(self.googlenet.children())[:-2]).to(self.device)
        self.video_list = []
        self.dataset = {}
        self.origin_h5 = h5py.File(origin_h5, 'r')
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
            self.files_list = os.listdir(video_path)
            self.video_list = [file for file in self.files_list if file.endswith((".avi",".flv",".mp4"))]

        for idx, file_name in enumerate(self.video_list):
            self.dataset[f'video_{idx+1}'] = {}
            self.h5_file.create_group(f'video_{idx+1}')
    
    def _extract_feature(self, frame): # frame's shape=(H,W,C)
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB
        im = Image.fromarray(im) # cv2 to PIL
        im = self.preprocess(im)
        im = im.unsqueeze(0).to(self.device) # it should be shape : (1,3,224,224)
        
        with torch.no_grad():
            feature = self.extractor(im).cpu().numpy().flatten() # [1(N), 1024, 1, 1] -> [1024]
    
        return feature
           
    def generate(self):
        for idx, file_name in enumerate(self.video_list):
            video_path = file_name
            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, file_name)
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) # for fps // 2 frame sampling
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            for i,k in enumerate(self.origin_h5.keys()): # origin tvsum don't has video name feature. so need to match by n_frames
                o_n_frames = self.origin_h5[k]['n_frames'][...]    
                if o_n_frames == n_frames or o_n_frames == n_frames-1: # for video_16 has 9534 but original is 9535 
                    print(f'{video_path}(FPS:{fps}) : {k} matched')
                    o = self.origin_h5[k]
                    idx = int(re.sub(r'[^0-9]', '', k))
                    break
                if i >= len(self.origin_h5.keys())-1:
                    print("there is no origin matching video")
            
            picks = []
            feature_stack_for_train = None
            
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

                else:
                    break    
            cap.release()
            
            # print("getting auto change points by KTS")
            # cps, n_frame_per_seg = self._get_change_points(feature_stack_for_cps, n_frames, fps)
            
            self.h5_file['video_{}'.format(idx)]['features'] = np.array(feature_stack_for_train)
            self.h5_file['video_{}'.format(idx)]['picks'] = np.array(list(picks))
            self.h5_file['video_{}'.format(idx)]['n_frames'] = n_frames
            self.h5_file['video_{}'.format(idx)]['change_points'] = o['change_points'][...]
            self.h5_file['video_{}'.format(idx)]['gtscore'] = o['gtscore'][...]
            self.h5_file['video_{}'.format(idx)]['gtsummary'] = o['gtsummary'][...]
            self.h5_file['video_{}'.format(idx)]['n_frame_per_seg'] = o['n_frame_per_seg'][...]
            self.h5_file['video_{}'.format(idx)]['n_steps'] = o['n_steps'][...]
            self.h5_file['video_{}'.format(idx)]['user_summary'] = o['user_summary'][...]
            #self.h5_file['video_{}'.format(idx+1)]['video_name'] = self.origin_h5[f'video_{idx+1}']['video_name'][...]

if __name__ == '__main__':
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input.split(os.path.sep)[-1]+'.h5'
    gen = Generate_Dataset(args.origin, args.input, args.output)
    gen.generate()
    gen.h5_file.close()
    
                    