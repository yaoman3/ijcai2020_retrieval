"""
Extract 3d feature
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import pdb
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.level2_dic = 'level2_dic_v1.npy'
    opt.level3_dic = 'level3_dic_v1.npy'
    opt.dataset_name = '3d_validation_set.npy' 
    opt.inplanes = 64
    opt.reverse = True
    opt.pose_num = 12
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    opt.phase = 'eval'
    
    opt.result_query_dir = 'extract_workshop_baseline_notexture_3d_v1'

    if not os.path.exists(opt.result_query_dir):
        os.makedirs(opt.result_query_dir)

    model.eval()
    for i, data in enumerate(dataset):
        model.set_input_eval(data)
        shape_id = data['shape_id'][0]
        # image_name = data['image_name'][0]
        
        # 3d
        # result_query_file = os.path.join(opt.result_query_dir, shape_id + '_' + image_name + '.npy')
        result_query_file = os.path.join(opt.result_query_dir, shape_id + '.npy')

        query_feat = model.retrieval_eval('3D')[-1]
        feat = {'feat_query':query_feat.cpu().detach().numpy(), 'shape_id':data['shape_id'][0]}
        
        np.save(result_query_file, feat)
        
