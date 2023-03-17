# **TPAMI2023 : Co-Salient Object Detection with Co-Representation Purification**

This repository is the official PyTorch implementation of our CoRP. [[**arXiv**](https://arxiv.org/abs/2303.07670)]

<div align=center><img width="550" height="230" src=./figures/main1.png/></div>

## **Abstract**

Co-salient object detection (Co-SOD) aims at discovering the common objects in a group of relevant images. Mining a co-representation is essential for locating co-salient objects. Unfortunately, the current Co-SOD method does not pay enough attention that the information not related to the co-salient object is included in the co-representation. Such irrelevant information in the co-representation interferes with its locating of co-salient objects.
In this paper, we propose a Co-Representation Purification (CoRP) method aiming at searching noise-free co-representation. We search a few pixel-wise embeddings probably belonging to co-salient regions. These embeddings constitute our co-representation and guide our prediction. For obtaining purer co-representation, we use the prediction to iteratively reduce irrelevant embeddings in our co-representation. Experiments on three datasets demonstrate that our CoRP achieves state-of-the-art performances on the benchmark datasets.
Our source code is available at https://github.com/ZZY816/CoRP.

## **Framework Overview**

<div align=center><img width="750" height="330" src=./figures/framework.png/></div>

## **Results**

The predicted results of our model trained by COCO9k only is available at [google-drive](https://drive.google.com/file/d/1YWxLQhe26bvFXfXzXIFw19mx69ESs1Lq/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/19sDWXHk0D04IlNdeGhdKDw) (fetch code: 7lmh)
+ quantitative results
<div align=center><img width="800" height="280" src=./figures/compare.png/></div>

+ qualitative results
<div align=center><img width="800" height="500" src=./figures/qualitative.png/></div>

## **Usage**
1. **Environment**

    ```
    Python==3.8.5
    opencv-python==4.5.3.56
    torch==1.9.0
    ```

2. **Datasets preparation**

    Download all the train/test datasets from my [google-drive](https://drive.google.com/file/d/1xD9BfxFnBl6vw0X97GXqLd8yBVR1tc3S/view?usp=sharing) and [google-drive](https://drive.google.com/file/d/1LAPmlWhnND9tBO3n_RaW2_ZIY0Jy1BGJ/view?usp=sharing), or [BaiduYun](https://pan.baidu.com/s/1npN6__inOd6uwKwza2TdZQ) (fetch code: s5m4). The file directory structure is as follows:
    ```
    +-- CoRP
    |   +-- Dataset
    |       +-- COCO9213  (Training Dataset for co-saliency branch)
    |       +-- Jigsaw_DUTS (Training Dataset for co-saliency branch)   
    |       +-- DUTS-TR (Training Dataset for saliency head)   
    |       +-- COCOSAL (Training Dataset for saliency head)  
    |       +-- CoSal2015 (Testing Dataset)   
    |       +-- CoCA (Testing Dataset)  
    |       +-- CoSOD3k (Testing Dataset)   
    |   +-- ckpt (The root for saving your checkpoint)
    |   ... 
    ```
 3. **Test and evalutation**
 
       Download the ckeckpoints of our model from [google-drive](https://drive.google.com/file/d/1viHjcuH0Ski67_zkgsQxAEhKL0Yf8Av8/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1YkWOjFNbtPjZs0VhROpZTA) (fetch code: utef). Place the **ckpt** folder in the main directory. Here is a command example of testing our model (trained by COCO9k with vgg16 backbone).
    ```
    CUDA_VISIBLE_DEVICES=0 python test.py --backbone vgg16 --ckpt_path './ckpt/vgg16_COCO9k/checkpoint.pth' --pred_root './Predictions/pred_vgg_coco/pred' 
    ```
    
    Run the following command to evaluate your prediction results，the metrics include **max F-measure**, **S-measure**, and **MAE**.
    
    ```
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --pred_root './Predictions/pred_vgg_coco/pred'
    ```
    For more metrics, CoSOD evaluation toolbox [eval-co-sod](https://github.com/zzhanghub/eval-co-sod) is strongly recommended.
    
 4. **Train your own model**

    Our CoRP can be trained with various backbones and training datasets.  
    ```
    CUDA_VISIBLE_DEVICES=0 python train.py --backbone <vgg16 or resnet50> --cosal_set <COCO9k or DUTS>  --sal_set <COCO9k or DUTS> --ckpt_root <Path for saving your checkpoint>
    ```
 
 ## Citation
  ```
  @ARTICLE{10008072,
  author={Zhu, Ziyue and Zhang, Zhao and Lin, Zheng and Sun, Xing and Cheng, Ming-Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Co-Salient Object Detection with Co-Representation Purification}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TPAMI.2023.3234586}}
  ```
 
 ## Contact
   
Feel free to leave issues here or send me e-mails (zhuziyue@mail.nankai.edu.cn).
