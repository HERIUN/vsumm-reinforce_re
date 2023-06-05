# video_summary_generaton
This repo contains the Pytorch implementation of the AAAI'18 paper - [Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward Summarization with Diversity-Representativeness Reward](https://arxiv.org/abs/1801.00054).

<div align="center">
  <img src="imgs/pipeline.jpg" alt="train" width="80%">
</div>

1. original repo is deprecated. re implement because some issues.

This code contains 
1) generate_h5_summe.py : making h5 file from original summe dataset : [my pytorch google net dataset](https://drive.google.com/file/d/1bOl551l-rsZ3A_LXjXz0b0rkNV05hBGk/view?usp=sharing)
1-1) generate_h5_tvsum.py : making h5 file from original tvsum dataset : [my pytorch google net dataset](https://drive.google.com/file/d/1ZWwJJpnhuE02Nao5dz0lXjvmcWBZkYs_/view?usp=sharing)
2) generate_dataset.py :  making h5 from video(.mp4)
3) create_split.py : split train,test videos
4) main.py : train DSN by h5 file
```
python main.py --dataset ~.h5 --split 
```
5) inference.py : making summarization video from custom video(not perfect)

If you want original video data.

[tvsum](http://people.csail.mit.edu/yalesong/tvsum/)

[summe](https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip)

Reference Repo

1) https://github.com/KaiyangZhou/pytorch-vsumm-reinforce
2) https://github.com/TorRient/Video-Summarization-Pytorch

Reference papers

1) Kaiyang Zhou,Yu Qia,Tao Xian.: "Reinforcement Learning for unsupervised video summarization with diversity-representativeness reward", arxiv:1801.00054v3[cs] , Feb.2018.
2) Tianrui Liu, Qingjie Meng, Athanasios Vlontzos, Jeremy Tan, DanielRueckert, Bernhard Kainz.:”Ultrasound Video Summarization usingDeep Reinforcement Learning”, arXiv:2005.09531 [cs], May. 2020.
3)  Danila Ptapov,Matthijs Douze, Zaid Harchaouni,Cordelia Schmid.:”Category-specific video summarization. ECCV-European conferenceon computer vision, Sep 2014,Zurich,Switzerland. pp.540-555,10.1007/978-3-319-10599-435.hal-01022967
5) Zhang,  K.,  Chao,  W.L.,  Sha,  F.,  Grauman,  K.:  Video  summarization  with  longshort-term  memory.  In:  European  conference  on  computer  vision.  pp.  766–782.Springer (2016)


