import os
import math
import random
from collections import defaultdict
import os.path as osp

import torchvision.transforms as transforms
import numpy as np
import re

from dassl.data.datasets import Datum, DatasetBase, DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CUB(DatasetBase):

    dataset_dir = 'cub'

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images_uncropped')
        self.anno_dir = os.path.join(self.dataset_dir, 'annotations')
        self.split_path = os.path.join(self.dataset_dir, f'split/splitv2/')
        

        self.template = template

        split = 'train'
        txt_path = os.path.join(self.split_path, f'{split}.csv')
        data, label = self.parse_csv(txt_path)

        train = self.convert_(data, label)

        split = 'test_seen'
        txt_path = os.path.join(self.split_path, f'{split}.csv')
        data, label = self.parse_csv(txt_path)

        test_seen = self.convert_(data, label)

        split = 'test_unseen'
        txt_path = os.path.join(self.split_path, f'{split}.csv')
        data, label = self.parse_csv(txt_path)

        test_unseen = self.convert_(data, label)

        self.train_classes = self.get_lab2cname(train)
        self.test_seen_classes = self.get_lab2cname(test_seen)
        self.test_unseen_classes = self.get_lab2cname(test_unseen)

        num_shots = cfg.DATASET.NUM_SHOTS

        # print('data', data)
        # print('label', label)

        # print('train ', train)
        # print([type(train), len(train)])
        # asd
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        # make base and new split here for co- co op
        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        if subsample == 'base':
            test = test_seen
        elif subsample == 'new': 
            test = test_unseen
            train = test # because classnames are used from train_x in the train
        else:
            ## this is only for zs
            # print(test_seen)
            # asd

            test = self._combine(test_seen, test_unseen)
            train = test ## hack to make it work because in this version of cub there is no train_unseen


            # print([len(test_seen), len(test_unseen), len(test)])
            # print(test_seen[-1].label)
            # print(test_unseen[-1].label)
            # print(test[-1].label)
            # asd

        super().__init__(train_x=train, val=test, test=test)
    


    
    def _combine(self, test_seen, test_unseen):
        # append max label number for test_seen to test_unseen
        # so that the labels are not overlapping
        max_label = max([x.label for x in test_seen])
        test_unseen_new = []
        for x in test_unseen:
            test_unseen_new.append(Datum(impath=x.impath, 
                                         label=x.label + max_label + 1, 
                                         classname=x.classname))
            # x.label += max_label + 1
        
        return test_seen + test_unseen_new


    @staticmethod
    def convert_(data, label):
        out = []
        for d, l in zip(data, label):
            impath = d
            label = l
            classname = ids2classes([os.path.basename(impath)])[0][:-len('.jpg')]
        
            
            item = Datum(
            impath=impath,
            label=int(label),
            classname=classname)

            out.append(item)
        return out

    def parse_csv(self, txt_path):
        data = []
        label = []
        lb = -1
        self.wnids = []
        # lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
        # new CSV does not have header
        lines = [x.strip() for x in open(txt_path, 'r').readlines()]
        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1].lower().split('.')[1] # uses label name not label number
            path = osp.join(self.image_dir, name)
            # print('wnid', wnid)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                
            data.append(path)
            label.append(lb)

        return data, label

template = ['a photo of a {}, a type of bird.']


def ids2classes(ids):

    if '_' in ids[0]: # Mini_Imagenet does not contain underscore, CUB does
        
        classes = np.array([re.sub('_\d+_\d+', '', id_) for id_ in ids])
    else:
        classes = np.array([id_[:-len('00000005')] for id_ in ids])
    return classes
