import os
from solver import Solver

"""
Test settings (used for "test.py"):

test_device:
    Index of the GPU used for test.

test_batch_size:
    Test batchsize.
  * When "test_batch_size == None", the dataloader takes the whole image group as a batch to
    perform the test (regardless of the size of the image group). If your GPU does not have enough memory,
    you are suggested to set "test_batch_size" with a small number (e.g. test_batch_size = 10).

pred_root:
    Folder path for saving predictions (co-saliency maps).

ckpt_path:
    Path of the checkpoint file (".pth") loaded for test.

original_size:
    When "original_size == True", the prediction (224*224) of ICNet will be resized to the original size.

test_roots:
    A dictionary including multiple sub-dictionary,
    each sub-dictionary contains the image and SISM folder paths of a specific test dataset.
    Format:
    test_roots = {
        name of dataset_1: {
            'img': image folder path of dataset_1,
            'sism': SISM folder path of dataset_1
        },
        name of dataset_2: {
            'img': image folder path of dataset_2,
            'sism': SISM folder path of dataset_2
        }
        .
        .
        .
    }
"""

test_device = '1'
test_batch_size = 40
pred_root = './pred_4_hw_reference/pred_160/'
pred_root_2 = './pred_4pre_location/pred_49_loc_final/'
ckpt_path = './ckpt/ckpt_4_hw_reference/Weights_160.pth'
original_size = False
test_num_thread = 4

# An example to build "test_roots".
test_roots = dict()
datasets = [  'hw_te']

for dataset in datasets:
    roots = {'img': './Dataset/{}/img/'.format(dataset),
             'gt': './Dataset/{}/gt/'.format(dataset),

             'sism': './Dataset/{}/sal/'.format(dataset)}
    test_roots[dataset] = roots
# ------------- end -------------

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = test_device
    solver = Solver()
    solver.test(roots=test_roots,
                ckpt_path=ckpt_path,
                pred_root=pred_root,

                num_thread=test_num_thread, 
                batch_size=test_batch_size, 
                original_size=original_size,
                pin=False)
