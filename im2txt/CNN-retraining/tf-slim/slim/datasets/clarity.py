# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the clarity dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/download_and_convert_flowers.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'clarity_%s_*.tfrecord'

SPLITS_TO_SIZES = {} # dictionary mapping the split name to the number of elements in that split; i.e. "train" : 3000, "validation" : 400

_NUM_CLASSES = None # even though this is supposed to be a constant, we have to define it at runtime since data_dir is only known when get_split is called

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and _NUM_CLASSES - 1 (which depends on which dataset is being trained)',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    
  def count(path): # count the number of files pseudo-recursively (since it only goes down one layer) in a directory
      num_files = 0
      
      for f in os.listdir(path):
          num_files += len(os.listdir(os.path.join(path,f)))
      
      return num_files
    
  """Gets a dataset tuple with instructions for reading flowers.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """

  if not os.path.exists(dataset_dir):
      print("Error: " + dataset_dir + " is not a valid path.")
      exit()
      
  train_dir = os.path.join(dataset_dir, "train")
  val_dir = os.path.join(dataset_dir, "val")
  test_dir = os.path.join(dataset_dir, "test")
      
  if not os.path.exists(train_dir):
      print("Error: " + train_dir + " is not a valid path.")
      exit()
      
  if not os.path.exists(val_dir):
      print("Error: " + val_dir + " is not a valid path.")
      exit()
      
  if not os.path.exists(test_dir):
      print("Error: " + test_dir + " is not a valid path.")
      exit()
      
      
  # Populate SPLITS_TO_SIZES based on the counts in the train, test, and val directories in data_dir
  
  # Note: SPLITS_TO_SIZES is of the form {"train" : 3000, "validation" : 400}
  
  # even though SPLITS_TO_SIZES is supposed to be a constant, we populate it in this function because this is
  # the only place where we have access to the dataset_dir directory
  
  _NUM_CLASSES = len(os.listdir(train_dir))
  
  SPLITS_TO_SIZES["train"] = count(train_dir)
  
  SPLITS_TO_SIZES["validation"] = count(val_dir)
  
  SPLITS_TO_SIZES["test"] = count(test_dir)
  
  print("HERE'S A SANITY CHECK: " + str(SPLITS_TO_SIZES))
  
  
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=_NUM_CLASSES,
      labels_to_names=labels_to_names)
