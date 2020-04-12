import os
import os.path as osp
import sys
import glob
import numpy as np
import random
import cv2

from datasets.data_loader import DataLoader


class MapsDataLoader(DataLoader):
    def __init__(self, dirnames=['out', 'osm'], **kwargs):
        dir_path = osp.dirname(osp.realpath(__file__))
        self.root_dir = osp.join(dir_path, 'data')
        self.dirnames = dirnames
        self.src_dirs = [osp.join(self.root_dir, s) for s in dirnames]
        
        # image filenames grouped by source directory
        self.img_filenames = [glob.glob(osp.join(s, '*.jpg')) \
                              for s in self.src_dirs]
        
        # image filenames as a flat list
        self.all_filenames = [x for l in self.img_filenames for x in l]
        
        # helper functions to map between key and path name
        self.path2key_fn = lambda s: s.split('/')[-1][4:-4]
        self.key2path_fn = lambda s, d: osp.join(self.root_dir, d,
                                                 d + '_' + s + '.jpg')
        # image keys grouped by source directory
        self.img_keys = [set(map(self.path2key_fn, filenames)) \
                         for filenames in self.img_filenames]
        
        # common keys across all source directories
        self.common_keys = set.intersection(*self.img_keys)
    
    def sample(self, n_samples):
        positive_key = random.sample(list(self.common_keys), k=1)[0]
        positive_dirs = random.sample(self.dirnames, k=2)
        positive_paths = [self.key2path_fn(positive_key, d) \
                          for d in positive_dirs]
       
        # sample negative samples in the same directory that the positive one
        # is sampled
        dir_index = self.dirnames.index(positive_dirs[1])
        negative_paths = random.sample(self.img_filenames[dir_index],
                                       k=n_samples - 1)
        
        all_paths = positive_paths + negative_paths
        
        # randomize position of positive item
        label = np.random.randint(n_samples)
        all_paths[1], all_paths[label + 1] = all_paths[label + 1], all_paths[1]
        
        # [n_samples + 1, h, w, c]
        try:
            images = np.stack([cv2.imread(path) for path in all_paths])
        except Exception:
            raise Exception('problem reading images')
        images = images.astype(float) / 255
        
        # setup return
        reference_image = images[0]
        target_images = images[1:]
        
        return reference_image, target_images, label
    
    def sample_batch(self, batch_size, n_samples):
        reference_image_list, target_images_list, label_list = [], [], []
        
        for i in range(batch_size):
            while True:
                try:
                    reference_image, target_images, label = self.sample(n_samples)
                    break
                except Exception:
                    continue
            reference_image_list.append(reference_image)
            target_images_list.append(target_images)
            label_list.append(label)
        
        # [bs, h, w, c]
        reference_image_np = np.stack(reference_image_list)
        
        # [bs, n_samples, h, w, c]
        target_images_np = np.stack(target_images_list)
        
        # [bs]
        labels_np = np.stack(label_list).astype(int)
        
        return reference_image_np, target_images_np, labels_np
