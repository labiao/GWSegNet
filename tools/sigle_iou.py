import os
import shutil

import cv2
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon, ttest_rel
from tqdm import tqdm


def Iou(i1, i2):
    # 计算交并
    intersection = np.logical_and(i1, i2).sum()
    union = np.logical_or(i1, i2).sum()

    if union == 0:
        iou = 1.0 if intersection == 0 else 0.0  # 两者都为0视为完美重合
    else:
        iou = intersection / union
    return iou

def compute_IoU(root1, root_gt):
    ious1 = []  # 存储每一张图的IoU
    ious1_set = set()

    for line in os.listdir(root1):
        path1 = os.path.join(root1, line)
        path3 = os.path.join(root_gt, line)

        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(path3, cv2.IMREAD_GRAYSCALE)

        img1 = (img1 > 0).astype(np.uint8)
        gt = (gt > 0).astype(np.uint8)

        iou1 = Iou(img1, gt)

        if iou1 <= 0.1:
            new_line = "_".join(line.split('_')[:-2])
            ious1_set.add(new_line)
            shutil.move(os.path.join('../fig_results/vaihingen/urbanssf-s-ditaiv4-train_tp', line),
                        os.path.join('../fig_results/vaihingen/urbanssf-s-ditaiv4-train_10', line))

        ious1.append(iou1)

    ious1_list = list(ious1_set)
    ious1_list = sorted(ious1_list, key=lambda x: str(x))
    with open('../data/ditaiv4/name_train_0.1.txt', 'w') as test_file:
        for n in ious1_list:
            test_file.write(f"{n}\n")

    return ious1

if __name__ == '__main__':

    root_gt = r'../data/ditaiv4/train/masks'

    ious1 = compute_IoU(r'../fig_results/vaihingen/urbanssf-s-ditaiv4-train', root_gt)

    bins = np.linspace(0.0, 1.0, 11)

    plt.figure(figsize=(6, 4))
    plt.hist([ious1], bins=bins, label=['iou'], stacked=False, alpha=0.7)
    plt.xlabel("IoU(%)")
    plt.ylabel("Number of Sample Pairs $(x_1, x_2, O)$")
    # plt.title("Distribution of Pseudo-label")
    # plt.xticks(np.linspace(0.4, 1.0, 13))

    # xtick_labels = [f"{int(bins[i]*100):.2f}–{int(bins[i + 1]*100):.2f}" for i in range(len(bins) - 1)]
    xtick_labels = [f"{int(bins[i]*100)}–{int(bins[i + 1]*100)}" for i in range(len(bins) - 1)]
    xtick_positions = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    plt.xticks(xtick_positions, xtick_labels, rotation=45)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join('../fig_results', 'Distribution_t1.png'), bbox_inches='tight', pad_inches=0.1)