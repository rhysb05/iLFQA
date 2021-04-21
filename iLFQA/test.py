import os
import sys
import time
from eli5_utils import qa_s2s_generate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import torch.nn as nn
from iLFQA import Loading_Model

#os.system("export PYTHONPATH=${PYTHONPATH}:`pwd`")

os.system("export TF_CPP_MIN_LOG_LEVEL=2")

res = Loading_Model()
data = input("Enter the question: ")
print(res.get_response(data))
