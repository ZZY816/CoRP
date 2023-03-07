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

+ quatitative results
<div align=center><img width="800" height="280" src=./figures/compare.png/></div>

+ qualitative results
<div align=center><img width="800" height="400" src=./figures/qualitative.png/></div>

