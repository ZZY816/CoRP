import os
from solver import Solver

"""
Training settings (used for "train.py"):

vgg_path:
    Path of pre-trained VGG16 (".pth") used to initialize ICNet at the start of training.

ckpt_root:
    Folder path where the checkpoint files (".pth") are saved.
    After the i-th training epoch, the checkpoint file is saved to "ckpt_root/Weights_{}.pth".format(i).

train_init_epoch:
    The starting epoch of training.
    When "train_init_epoch == 0", ICNet is initialized with pre-trained VGG16;
    Otherwise, ICNet loads checkpoint file from "ckpt_root/Weights_{}.pth".format(train_init_epoch) for initialization,

train_end_epoch:
    The ending epoch of training.
    We recommend you to train ICNet for 50~60 epochs.

train_device:
    Index of the GPU used for training.

train_doc_path:
    The file (".txt") path used to save the training information.

train_roots:
    A dictionary containing image, GT and SISM folder paths of the training dataset.
    train_roots = {'img': image folder path of training dataset,
                   'gt': GT folder path of training dataset,
                   'sism': SISM folder path of training dataset}
"""
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

vgg_path = './CoRP_backnone.pth'

ckpt_root = './ckpt/ckpt_4_hw_reference/'
train_init_epoch = 0
train_end_epoch = 200
train_device = '0'
train_doc_path = './training_4_hw_reference.txt'
learning_rate = 1e-5
weight_decay = 1e-4
train_batch_size = 10
train_num_thread = 1

# An example to build "train_roots".
train_roots = {'img': './Dataset/hw_tr/img/',
               'gt': './Dataset/hw_tr/gt/',
               'sism': './Dataset/hw_tr/sal/'}
# ------------- end -------------

if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = train_device
    solver = Solver()
    solver.train(roots=train_roots,
                 vgg_path=vgg_path,
                 init_epoch=train_init_epoch,
                 end_epoch=train_end_epoch,
                 learning_rate=learning_rate,
                 batch_size=train_batch_size,
                 weight_decay=weight_decay,
                 ckpt_root=ckpt_root,
                 doc_path=train_doc_path,
                 num_thread=train_num_thread,
                 pin=False)
