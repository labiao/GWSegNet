import os
import shutil

import numpy as np
from PIL import Image

root_img_512 = r'../data/ditaiv4/test/images'
root_512 = r'../data/ditaiv4/test/masks'
root_ori = r'../data/ditaiv4/masks_oriv4'

for n in os.listdir(root_512):
	path_512 = os.path.join(root_512, n)
	path_img_512 = os.path.join(root_img_512, n.replace('png', 'tif'))
	new_n = "_".join(n.split('_')[:-2]) + '.png'
	path_ori = os.path.join(root_ori, new_n)

	mask_512 = np.array(Image.open(path_512))
	ori_512 = np.array(Image.open(path_ori))

	bin_mask_512 = (mask_512 == 1).astype(np.uint8)
	bin_ori_512 = (ori_512 == 1).astype(np.uint8)

	# 连通域分析
	area_in_crop = np.sum(bin_mask_512)
	original_area = np.sum(bin_ori_512)
	if original_area != 0 and area_in_crop != original_area:
		os.remove(path_512)
		os.remove(path_img_512)
	# if area_in_crop != original_area:
	#     os.remove(path_512)
