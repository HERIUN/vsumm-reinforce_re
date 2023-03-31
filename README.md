# video_summary_generaton
This repo contains the Pytorch implementation of the AAAI'18 paper - Deep Reinforcement Learning for Unsupervised Video
Summarization with Diversity-Representativeness Reward.


1. The original repo is deprecated. I reinplement it(~ing)
This code contains 
1) generate_h5_google_summe.py : making h5 file from original summe dataset
2) generate_dataset.py :  making h5 from video(.mp4)
3) main.py : train DSN by h5 file
4) inference.py : making summarization video from custom video(not perfect)

Reference Repo

1) https://github.com/KaiyangZhou/pytorch-vsumm-reinforce
2) https://github.com/TorRient/Video-Summarization-Pytorch

Reference papers

1) Kaiyang Zhou,Yu Qia,Tao Xian.: "Reinforcement Learning for unsupervised video summarization with diversity-representativeness reward", arxiv:1801.00054v3[cs] , Feb.2018.
2) Tianrui Liu, Qingjie Meng, Athanasios Vlontzos, Jeremy Tan, DanielRueckert, Bernhard Kainz.:”Ultrasound Video Summarization usingDeep Reinforcement Learning”, arXiv:2005.09531 [cs], May. 2020.
3)  Danila Ptapov,Matthijs Douze, Zaid Harchaouni,Cordelia Schmid.:”Category-specific video summarization. ECCV-European conferenceon computer vision, Sep 2014,Zurich,Switzerland. pp.540-555,10.1007/978-3-319-10599-435.hal-01022967
5) Zhang,  K.,  Chao,  W.L.,  Sha,  F.,  Grauman,  K.:  Video  summarization  with  longshort-term  memory.  In:  European  conference  on  computer  vision.  pp.  766–782.Springer (2016)


