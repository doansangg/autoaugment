# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Data utils for CIFAR-10 and CIFAR-100."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import pickle as cPickle
import os
import augmentation_transforms
import numpy as np
import policies as found_policies
import tensorflow as tf
from PIL import Image
#load data train,val,test
#ration train:val =0.2
def LoadPathDATA(path_train,path_test,width,height,ration,n_class):
  f_train=open(path_train,'r')
  f_test=open(path_test,'r')
  list_train=f_train.readlines()
  list_test=f_test.readlines()
  images_train=[]
  labels_train=[]
  images_test=[]
  labels_test=[]
  images_val=[]
  labels_val=[]
  for i,p in enumerate(list_train):
    path_image = p.split("\t")[0]
    label_path = int(p.split("\t")[1].split('\n')[0])
    length=len(list_train)
    image = Image.open(path_image)
    image = image.resize((width, height), Image.ANTIALIAS)
    imgarr = np.array(image,dtype=np.uint8)
    if i >=100:
      images_train.append(imgarr)
      labels_train.append(label_path)
    if i<100:
      images_val.append(imgarr)
      labels_val.append(label_path)
  for i,p in enumerate(list_test):
    path_image = p.split("\t")[0]
    label_path = int(p.split("\t")[1].split('\n')[0])
    image = Image.open(path_image)
    image = image.resize((width, height), Image.ANTIALIAS)
    imgarr = np.array(image,dtype=np.uint8)
    images_test.append(imgarr)
    labels_test.append(label_path)
  train_img= np.array(images_train)
  print('shape train: ',train_img.shape)
  train_lb = np.eye(n_class)[np.array(labels_train, dtype=np.int32)]
  val_img= np.array(images_val) 
  val_lb = np.eye(n_class)[np.array(labels_val, dtype=np.int32)]
  test_img= np.array(images_test) 
  test_lb = np.eye(n_class)[np.array(labels_test, dtype=np.int32)]
  #process data train
  train_img = train_img / 255.0
  mean = augmentation_transforms.MEANS
  std = augmentation_transforms.STDS
  train_img = (train_img - mean) / std

  #process data val
  val_img = val_img / 255.0
  mean = augmentation_transforms.MEANS
  std = augmentation_transforms.STDS
  val_img = (val_img - mean) / std

  #process data test
  test_img = test_img / 255.0
  mean = augmentation_transforms.MEANS
  std = augmentation_transforms.STDS
  test_img = (val_img - mean) / std
  return train_img,train_lb,val_img,val_lb,test_img,test_lb



class DataSet(object):
  """Dataset object that produces augmented training and eval data."""

  def __init__(self, hparams):
    self.hparams = hparams
    self.epochs = 0
    self.curr_train_index = 0
    
    self.good_policies = found_policies.good_policies()
    #number class
    self.n_class=self.hparams.n_class
    self.size_width=self.hparams.width
    self.size_heigth=self.hparams.heigth
    self.ration=self.hparams.ration
    #path folder train
    self.path_train=self.hparams.path_train
    self.path_test=self.hparams.path_test
    #load data
    

    self.train_images,self.train_labels,self.val_images,self.val_labels,self.test_images,self.test_labels =  LoadPathDATA(self.path_train,self.path_test,self.size_width,self.size_heigth,self.ration,self.n_class)
    self.num_train = self.train_images.shape[0]

  def next_batch(self):
    """Return the next minibatch of augmented data."""
    next_train_index = self.curr_train_index + self.hparams.batch_size
    if next_train_index > self.num_train:
      # Increase epoch number
      epoch = self.epochs + 1
      self.reset()
      self.epochs = epoch
    batched_data = (
        self.train_images[self.curr_train_index:
                          self.curr_train_index + self.hparams.batch_size],
        self.train_labels[self.curr_train_index:
                          self.curr_train_index + self.hparams.batch_size])
    final_imgs = []

    images, labels = batched_data
    print(images.shape)
    for data in images:
      epoch_policy = self.good_policies[np.random.choice(
          len(self.good_policies))]
      final_img = augmentation_transforms.apply_policy(
          epoch_policy, data)
      final_img = augmentation_transforms.random_flip(
          augmentation_transforms.zero_pad_and_crop(final_img, 4))
      # Apply cutout
      final_img = augmentation_transforms.cutout_numpy(final_img)
      final_imgs.append(final_img)
    batched_data = (np.array(final_imgs, np.float32), labels)
    self.curr_train_index += self.hparams.batch_size
    return batched_data

  def reset(self):
    """Reset training data and index into the training data."""
    self.epochs = 0
    # Shuffle the training data
    perm = np.arange(self.num_train)
    np.random.shuffle(perm)
    self.train_images = self.train_images[perm]
    self.train_labels = self.train_labels[perm]
    self.curr_train_index = 0
