# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import json
import time

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)

# main will write out a json file
def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    ### altered code to make a results json file ###
    results = []
    for filename in filenames:
      results_entry = {}
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      results_entry["image_id"] = filename[-11:-4]
      # change caption index to see a different caption
      # 0 = top result, 1 = 2nd best result, etc...
      sentence = [vocab.id_to_word(w) for w in captions[0].sentence[1:-1]]
      sentence = " ".join(sentence)
      results_entry["caption"] = sentence
      results.append(results_entry)
    with open('/scratch/kpmoran/Clarity/im2txt/data/results/results.json', 'w') as outfile:
      json.dump(results, outfile)
      
      
# inference on a specific checkpoint, return a JSON with captions
# linked to image ids
def inference_on_ckpt(ckpt_path, vocab_file, input_files):
    
    #if not os.path.isfile(ckpt_path):
        #return None
    
    #tf.app.run()

    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   ckpt_path)
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(vocab_file)

    filenames = []
    for file_pattern in input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    tf.logging.info("Running caption generation on %d files matching %s",
                                        len(filenames), input_files)
                                        
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
  
  
    # create a session that only uses a tenth of the available GPU memory
    with tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        #i=0
        
        # use beam size of 3, take top 3 captions from predictive model
        BEAM_SIZE = 3
        
        generator = caption_generator.CaptionGenerator(model, vocab, beam_size=BEAM_SIZE)
        ### altered code to make a results json file ###
        results = []
        for filename in filenames:
            start = time.time()
            
            #i+=1
            #print("\n#%d INFERENCING ON "%(i) + filename, end = " ")
            results_entry = {}
            with tf.gfile.GFile(filename, "rb") as f:
                
                image = f.read()
                start=time.time()
                captions = generator.beam_search(sess, image)
                #print(captions)
                results_entry["image_id"] = filename[-11:-4]
                # change caption index to see a different caption
                # 0 = top result, 1 = 2nd best result, etc...
                
                # list of captions as strings (we call them sentences here)
                sentences = []
                
                for i in range(BEAM_SIZE): # 0 to 2 inclusive
                    
                    # if i is in range of the generated captions
                    if (i < len(captions)):
                        sentence = [vocab.id_to_word(w) for w in captions[i].sentence[1:-1]]
                        sentence = " ".join(sentence)
                    
                        sentences.append(sentence)
                    else:
                         sentences.append("") # append an empty string
                         print("Could not fetch caption #%d"%i)
                         
                
                
                
                results_entry["captions"] = sentences
                results.append(results_entry)
                #print(str(time.time() - start) + " s")
            print(filename + " in %s s"%str(time.time()-start))
        
    # return all the predictions
    return results


if __name__ == "__main__":
  tf.app.run()

