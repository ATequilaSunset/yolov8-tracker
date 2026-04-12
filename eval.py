"""py-motmetrics - metrics for multiple object tracker (MOT) benchmarking.
Christoph Heindl, 2017
https://github.com/cheind/py-motmetrics
Modified by Xingyi Zhou
"""

import argparse
import logging
import numpy as np

if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm
from collections import OrderedDict
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="""
Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt.txt
    ...
Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>/<SEQUENCE_2>.txt
    ...
Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--gt-root', type=str, default='/data/lz/SatVideoDT/val_data')
    parser.add_argument('--result', type=str, default='./track_results_vis/sort_bytetrack')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot16')
    parser.add_argument('--solver', type=str, help='LAP solver to use')
    return parser.parse_args()

def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.7))
            names.append(k)
        else:
            logging.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names


def collect_gt_files(gt_root):
    return sorted(str(path) for path in Path(gt_root).glob('*/gt.txt'))


def collect_result_files(result_root):
    return sorted(str(path) for path in Path(result_root).glob('*/*.txt'))

if __name__ == '__main__':

    args = parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))        
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    gtfiles = collect_gt_files(args.gt_root)
    print('gt_files', gtfiles)
    tsfiles = collect_result_files(args.result)

    logging.info('Found {} groundtruths and {} test files.'.format(len(gtfiles), len(tsfiles)))
    logging.info('Available LAP solvers {}'.format(mm.lap.available_solvers))
    logging.info('Default LAP solver \'{}\''.format(mm.lap.default_solver))
    logging.info('Loading files.')
    
    gt = OrderedDict([(Path(f).parts[-2], mm.io.loadtxt(f, fmt=args.fmt)) for f in gtfiles])
    ts = OrderedDict([(Path(f).parts[-2], mm.io.loadtxt(f, fmt=args.fmt)) for f in tsfiles])

    mh = mm.metrics.create()    
    accs, names = compare_dataframes(gt, ts)
    
    logging.info('Running metrics')
    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked', \
      'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', \
      'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(
      accs, names=names, 
      metrics=metrics, generate_overall=True)
    # summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    # print(mm.io.render_summary(
    #   summary, formatters=mh.formatters, 
    #   namemap=mm.io.motchallenge_metric_names))
    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses', 
          'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked', 
          'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches', 
      'num_fragmentations', 'mostly_tracked', 'partially_tracked', 
      'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    # print(mm.io.render_summary(
    #   summary, formatters=fmt, 
    #   namemap=mm.io.motchallenge_metric_names))
    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(
    accs, names=names, 
    metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(
    summary, formatters=mh.formatters, 
    namemap=mm.io.motchallenge_metric_names))
    logging.info('Completed')

