# video_summary_generaton
This repo contains the Pytorch implementation of the AAAI'18 paper - Deep Reinforcement Learning for Unsupervised Video
Summarization with Diversity-Representativeness Reward.

The repo contains 2 files.
1) training.ipynb
2) testing.ipynb

Please assign the path of training data set to the variable 'input_videos_folder' in training.py.The program will preprocess the data and 
will generate model.

Please assign the path of testing data set to the variable 'input_videos_folder' in testing.py and provide path of the model generated during training.The summary will be generated in a separate folder 'summary_videos'.

For more details contact me:  anaghazac@gmail.com

Reference Repo

1) https://github.com/KaiyangZhou/pytorch-vsumm-reinforce
2) https://github.com/TorRient/Video-Summarization-Pytorch

Reference papers

1) Kaiyang Zhou,Yu Qia,Tao Xian.: "Reinforcement Learning for unsupervised video summarization with diversity-representativeness reward", arxiv:1801.00054v3[cs] , Feb.2018.
2) Tianrui Liu, Qingjie Meng, Athanasios Vlontzos, Jeremy Tan, DanielRueckert, Bernhard Kainz.:”Ultrasound Video Summarization usingDeep Reinforcement Learning”, arXiv:2005.09531 [cs], May. 2020.
3)  Danila Ptapov,Matthijs Douze, Zaid Harchaouni,Cordelia Schmid.:”Category-specific video summarization. ECCV-European conferenceon computer vision, Sep 2014,Zurich,Switzerland. pp.540-555,10.1007/978-3-319-10599-435.hal-01022967
5) Zhang,  K.,  Chao,  W.L.,  Sha,  F.,  Grauman,  K.:  Video  summarization  with  longshort-term  memory.  In:  European  conference  on  computer  vision.  pp.  766–782.Springer (2016)


