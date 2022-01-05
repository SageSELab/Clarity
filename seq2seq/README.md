[![CircleCI](https://circleci.com/gh/google/seq2seq.svg?style=svg)](https://circleci.com/gh/google/seq2seq)

---

**[READ THE DOCUMENTATION](https://google.github.io/seq2seq)**

**[CONTRIBUTING](https://google.github.io/seq2seq/contributing/)**

---

A general-purpose encoder-decoder framework for Tensorflow that can be used for Machine Translation, Text Summarization, Conversational Modeling, Image Captioning, and more.

![Translation Model](https://3.bp.blogspot.com/-3Pbj_dvt0Vo/V-qe-Nl6P5I/AAAAAAAABQc/z0_6WtVWtvARtMk0i9_AtLeyyGyV6AI4wCLcB/s1600/nmt-model-fast.gif)

---

# Instructions to run seq2seq on bg9 and Semeru Tower 2
To run seq2seq you will need to create a virtual environment with python, tensorflow 1.0, nltk, and the nltk 'punkt' library

conda create -n myenv

conda activate myenv

pip install tensorflow-gpu=1.0

conda install nltk

python
>import nltk

>nltk.download('punkt')

>quit()

# Preprocess 

There is a bash script in the seq2seq directiory that will execute the required preprocessing (preprocess_seq2seq.sh). To complete preprocessing:

* activate your virtual environment
* check the file paths in the preprocessing script to ensure accuracy
* run the preprocessing script(example below)

./preprocess_seq2seq.sh  

# Train

The training script (train_seq2seq.sh) is run with the same steps as the preprocessing script.

\* If you recieve the error "ImportError: libcudart.so.8.0: cannot open shared object file: No such file or directory" try:

* export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64

\* If you recieve the error "Couldn't open CUDA library libcupti.so.8.0. LD_LIBRARY_PATH: /usr/local/cuda-8.0/lib64" try:

* export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH


# Inference

Before running the inference script check that the environmental variables in the script match the environmental variables you used to train seq2seq.
The inference script has several options that you can change like UNK replacement. Further details can be found at `https://google.github.io/seq2seq/inference/`.

./inference_seq2seq.sh

# Evaluate

The evaluate script uses the multi-bleu.pearl script from `[Moses]https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl`. It 
currently prints results to stdout, but results can be redirected to a file using the '>' operator.

./evaluate_seq2seq.sh

---

The official code used for the [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906) paper.