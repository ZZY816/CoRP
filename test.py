import os
from solver import Solver
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default='vgg16',
                        help='The optional backbones are vgg16 and resnet50.'
                             'It should match the checkpoint.')

    parser.add_argument('--test_batch_size', type=int, default=None,
                        help='When "test_batch_size == None", the dataloader takes the whole image group as a batch to '
                             'perform the test (regardless of the size of the image group). If your GPU does not have '
                             'enough memory, you are suggested to set "test_batch_size" with a small number '
                             '(e.g. test_batch_size = 10).')

    parser.add_argument('--pred_root', type=str, default='./Predictions/pred_vgg/pred',
                        help='Folder path for saving predictions (co-saliency maps).')

    parser.add_argument('--ckpt_path', type=str, default='./ckpt/ckpt_bn2/Weights_1.pth',
                        help='Path of the checkpoint file (".pth") loaded for test.')

    parser.add_argument('--original_size', type=bool, default=True,
                        help='When "original_size == True", '
                             'the prediction (224*224) of ICNet will be resized to the original size..')

    parser.add_argument('--test_num_thread', type=int, default=4, help='num_thread.')

    parser.add_argument('--datasets', type=list, default=['CoCA', 'CoSal2015', 'CoSOD3k'], help='test dataset.')

    return parser.parse_args()



def main():
    args = parse_args()
    solver = Solver(backbone=args.backbone)

    # An example to build "test_roots".
    test_roots = dict()

    for dataset in args.datasets:
        roots = {'img': './Dataset/{}/img/'.format(dataset),
                 'gt': './Dataset/{}/gt/'.format(dataset),

                 }
        test_roots[dataset] = roots

    solver.test(roots=test_roots,
                ckpt_path=args.ckpt_path,
                pred_root=args.pred_root,
                num_thread=args.test_num_thread,
                batch_size=args.test_batch_size,
                original_size=args.original_size,
                pin=False)


# ------------- end -------------

if __name__ == '__main__':
    main()

