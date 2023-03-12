from evaluator.evaluator import evaluate_dataset
from utils import write_doc
import argparse

"""
* Note:
    The evaluation codes in "./evaluator/" are implemented in PyTorch (GPU-version) for acceleration.

    Since some GTs (e.g. in "Cosal2015" dataset) are of too large original sizes to be evaluated on GPU with limited memory 
    (our "RTX 2080ti" runs out of 12G memory when computing F-measure), the input prediction map and corresponding GT 
    are resized to 224*224 by our evaluation codes before computing metrics.
"""

"""
evaluate:
    Given predictions, compute multiple metrics (max F-measure, S-measure and MAE).
    The evaluation results are saved in "doc_path".
"""


def evaluate(roots, doc_path, num_thread, pin):
    datasets = roots.keys()
    for dataset in datasets:
        # Evaluate predictions of "dataset".
        results = evaluate_dataset(roots=roots[dataset], 
                                   dataset=dataset,
                                   batch_size=1, 
                                   num_thread=num_thread, 
                                   demical=True,
                                   suffixes={'gt': '.png', 'pred': '.png'},
                                   pin=pin)
        
        # Save evaluation results.
        content = '{}:\n'.format(dataset)
        content += 'mean-Fmeasure={}'.format(results['mean_f'])
        content += ' '
        content += 'max-Fmeasure={}'.format(results['max_f'])
        content += ' '
        content += 'Smeasure={}'.format(results['s'])
        content += ' '
        content += 'MAE={}\n'.format(results['mae'])
        write_doc(doc_path, content)
    content = '\n'
    write_doc(doc_path, content)


def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--pred_root', type=str, default='./Predictions/pred_vgg_coco/pred',
                        help='Folder path for the predictions')

    parser.add_argument('--eval_num_thread', type=int, default=4, help='Thread number.')

    parser.add_argument('--datasets', type=list, default=['CoCA', 'CoSal2015', 'CoSOD3k'], help='test dataset.')

    return parser.parse_args()


def main():
    args = parse_args()
    eval_doc_path = args.pred_root + '/evaluation.txt' # Txt Path to save the evaluation results.

    # An example to build "eval_roots".
    eval_roots = dict()
    for dataset in args.datasets:
        roots = {'gt': './Dataset/{}/gt/'.format(dataset),
                 'pred': args.pred_root + '/{}/'.format(dataset)}
        eval_roots[dataset] = roots
    eval_num_thread = args.eval_num_thread

    evaluate(roots=eval_roots,
             doc_path=eval_doc_path,
             num_thread=eval_num_thread,
             pin=False)


# ------------- end -------------

if __name__ == "__main__":
    main()