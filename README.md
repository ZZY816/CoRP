# **TPAMI2023 : Co-Salient Object Detection with Co-Representation Purification**

This repository is the official PyTorch implementation of our CoRP.

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
1. **Datasets preparation**

    Download all the train/test datasets from my [google-drive](https://drive.google.com/file/d/1xD9BfxFnBl6vw0X97GXqLd8yBVR1tc3S/view?usp=sharing) and [google-drive](https://drive.google.com/file/d/1LAPmlWhnND9tBO3n_RaW2_ZIY0Jy1BGJ/view?usp=sharing), or [BaiduYun](https://pan.baidu.com/s/1wOxdP6EQEqMwjg3_v1z2-A) (fetch code: 5183). The file directory structure is as follows:
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
 
