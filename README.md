# CoRP

## 训练和测试

### 准备 SISMs

CoRP 可以基于任意现成 SOD 方法产生的 SISMs 进行训练和测试，但我们建议您在训练和测试阶段使用**相同**的 SOD 方法来生成 SISMs，以确保在训练和测试时的一致性。


### 训练

1. 下载预训练的 VGG16：

   * ***vgg16_feat.pth*** (56MB)，[GoogleDrive](https://drive.google.com/file/d/1ej5ngj2NYH-R-0GfYUDfuM-DNLuFolED/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1kAh7FAUPuVLI5cvtBsxh-A) (提取码：j0zq)。

2. 根据 **"./CoRP/train.py"** 中的说明修改训练设置。

3. 运行：

```
python train.py
```

### 测试

1. * 测试**预训练的** CoRP：

     下载预训练的 CoRP ***"CoRP_vgg16.pth"*** (下载链接已在先前给出)。

   * 测试**您自己训练的** CoRP：

     选择您想要加载的检查点文件 ***"Weights_i.pth"***  (在第 i 个 epoch 训练后会自动保存)。

2. 根据 **"./CoRP/test.py"** 中的说明修改测试设置。

3. 运行：

```
python test.py
```

## 评测

文件夹 "./CoRP/evaluator/" 包含了用 PyTorch (GPU版本) 实现的评测代码，评测指标有 **max F-measure**、**S-measure** 以及 **MAE** 。

1. 根据 **"./CoRP/evaluate.py"** 中的说明修改评测设置。

2. 运行：

```
python evaluate.py
```
