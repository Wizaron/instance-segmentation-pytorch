import argparse
import numpy as np
from PIL import Image
import os

parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', required=True, help='Prediction directory')
parser.add_argument('--dataset', type=str,
                    help='Name of the dataset which is "CVPPP"',
                    required=True)
opt = parser.parse_args()

assert opt.dataset in ['CVPPP', ]

pred_dir = opt.pred_dir


def calc_dic(n_objects_gt, n_objects_pred):
    return np.abs(n_objects_gt - n_objects_pred)


def calc_dice(gt_seg, pred_seg):

    nom = 2 * np.sum(gt_seg * pred_seg)
    denom = np.sum(gt_seg) + np.sum(pred_seg)

    dice = float(nom) / float(denom)
    return dice


def calc_bd(ins_seg_gt, ins_seg_pred):

    gt_object_idxes = list(set(np.unique(ins_seg_gt)).difference([0]))
    pred_object_idxes = list(set(np.unique(ins_seg_pred)).difference([0]))

    best_dices = []
    for gt_idx in gt_object_idxes:
        _gt_seg = (ins_seg_gt == gt_idx).astype('bool')
        dices = []
        for pred_idx in pred_object_idxes:
            _pred_seg = (ins_seg_pred == pred_idx).astype('bool')

            dice = calc_dice(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    best_dice = np.mean(best_dices)

    return best_dice


def calc_sbd(ins_seg_gt, ins_seg_pred):

    _dice1 = calc_bd(ins_seg_gt, ins_seg_pred)
    _dice2 = calc_bd(ins_seg_pred, ins_seg_gt)
    return min(_dice1, _dice2)


if opt.dataset == 'CVPPP':
    names = np.loadtxt('../data/metadata/CVPPP/validation_image_paths.txt',
                       dtype='str', delimiter=',')
    names = np.array([os.path.splitext(os.path.basename(n))[0] for n in names])
    n_objects_gts = np.loadtxt(
        '../data/metadata/CVPPP/number_of_instances.txt',
        dtype='str',
        delimiter=',')
    img_dir = '../data/raw/CVPPP/CVPPP2017_LSC_training/training/A1'

    dics, sbds, fg_dices = [], [], []
    for name in names:
        if not os.path.isfile(
                '{}/{}/{}-n_objects.npy'.format(pred_dir, name, name)):
            continue

        n_objects_gt = int(n_objects_gts[n_objects_gts[:, 0] == name.replace('_rgb', '')][0][1])
        n_objects_pred = np.load(
            '{}/{}/{}-n_objects.npy'.format(pred_dir, name, name))

        ins_seg_gt = np.array(Image.open(
            os.path.join(img_dir, name.replace('_rgb', '') + '_label.png')))
        ins_seg_pred = np.array(Image.open(os.path.join(
            pred_dir, name, name + '-ins_mask.png')))

        fg_seg_gt = np.array(
            Image.open(
                os.path.join(
                    img_dir,
                    name.replace('_rgb', '') +
                    '_fg.png')))
        fg_seg_pred = np.array(Image.open(os.path.join(
            pred_dir, name, name + '-fg_mask.png')))

        fg_seg_gt = (fg_seg_gt == 1).astype('bool')
        fg_seg_pred = (fg_seg_pred == 255).astype('bool')

        sbd = calc_sbd(ins_seg_gt, ins_seg_pred)
        sbds.append(sbd)

        dic = calc_dic(n_objects_gt, n_objects_pred)
        dics.append(dic)

        fg_dice = calc_dice(fg_seg_gt, fg_seg_pred)
        fg_dices.append(fg_dice)

    mean_dic = np.mean(dics)
    mean_sbd = np.mean(sbds)
    mean_fg_dice = np.mean(fg_dices)

    print 'MEAN SBD     : ', mean_sbd
    print 'MEAN |DIC|   : ', mean_dic
    print 'MEAN FG DICE : ', mean_fg_dice
