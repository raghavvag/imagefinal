# -*- coding: utf-8 -*-
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.md
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

import os, torch
import glob
import time
from PIL import Image
from resnet50nodown import resnet50nodown
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    
    weights_path = 'models/ganimagedetection_image/weights/gandetection_resnet50nodown_stylegan2.pth'
    input_folder = './temp'
    output_csv = 'models/ganimagedetection_image/result.txt'
    
    from torch.cuda import is_available as is_available_cuda
    device = 'cuda:0' if is_available_cuda() else 'cpu'
    net = resnet50nodown(device, weights_path)
    
    list_files = sorted(sum([glob.glob(os.path.join(input_folder,'*.'+x)) for x in ['jpg','JPG','jpeg','JPEG','png','PNG']], list()))
    num_files = len(list_files)
    
    # print('GAN IMAGE DETECTION')
    # print('START')
    
    with open(output_csv, 'w') as fid:
        # fid.write('filename,logit,probability,time\n')
        fid.flush()
        for index, filename in enumerate(list_files):
            # print('%5d/%d' % (index, num_files), end='\r')
            tic = time.time()
            img = Image.open(filename).convert('RGB')
            img.load()
            logit = net.apply(img)
            probability = torch.sigmoid(torch.tensor(logit)).item()  # For binary classification
            toc = time.time()
            
            # fid.write('%s,%f,%f,%f\n' % (filename, logit, probability, toc-tic))
            fid.write('%f' % (probability))
            fid.flush()

    # print('\nDONE')
    # print('OUTPUT: %s' % output_csv)
