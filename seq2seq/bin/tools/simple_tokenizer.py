#! /usr/bin/env python

"""Tokenizes a text file """

import string
import nltk
import sys
import argparse

parser = argparse.ArgumentParser("Tokenizes a text file")
parser.add_argument(
  "--infile",
  dest="infile",
  type=str,
  help="File to be tokenized")
parser.add_argument(
  "--outfile",
  dest="outfile",
  type=str,
  help="Name of tokenized file")
args = parser.parse_args()

translator = str.maketrans('', '', string.punctuation)

with  open(args.infile, 'r') as f:
  file_content = f.read()
tokens = nltk.word_tokenize(file_content)
with open(args.outfile, 'w+') as f:
  for tok in tokens:
    f.write(tok.translate(translator) + '\n')

