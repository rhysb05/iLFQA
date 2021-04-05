from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from eli5_utils import qa_s2s_generate
from docqa.data_processing.text_utils import NltkAndPunctTokenizer, NltkPlusStopWords
from docqa.data_processing.document_splitter import MergeParagraphs
import re
from os.path import isfile
from iTA_BERT import Loading_Model
import time
import pandas as pd
import numpy as np
from nltk.translate import bleu_score
import collections, nltk

res = Loading_Model()

def unigram(tokens):    
    model = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        try:
            model[f] += 1
        except KeyError:
            model [f] = 1
            continue
    N = float(sum(model.values()))
    for word in model:
        model[word] = model[word]/N
    return model

def perplexity(testset, model):
    testset = testset.split()
    perplexity = 1
    N = 0
    for word in testset:
        N += 1
        perplexity = perplexity * (1/model[word])
    perplexity = pow(perplexity, 1/float(N)) 
    return perplexity

data_frame = pd.read_excel("/home/bdlabucdenver/data/QA_rephrase.xlsx")
np_data = np.array(data_frame)
records_count = 0
final_data = np.array([['Question', 'Answer', 'zero_shot_time', 'tf_idf_time', 'confidence_score_time', 'text_generation_time', 'Bleu', 'Perplex', 'G_answer']])
for d in np_data:
    start = time.time()
    answer, time_dict = res.get_response_BERT_answer_para_concat(d[0])
    print(answer)
    end = time.time()

    BLEU = bleu_score.sentence_bleu(d[1], answer)
    tokens = nltk.word_tokenize(d[1])
    model = unigram(tokens)
    perplex = perplexity(answer, model)
    all_values = [d[0], d[1], time_dict['zero_shot_time'],time_dict['tf_idf'], time_dict['confidence_scores'], time_dict['answer'], BLEU, perplex, answer]
    final_data = np.append(final_data, [all_values], axis = 0)
    records_count += 1
    print("Done {}\n".format(records_count))
df = pd.DataFrame(final_data)
df.to_excel("/home/bdlabucdenver/data/BERT_answer_para_concat_zero_shot.xlsx", index = False, header= False)
