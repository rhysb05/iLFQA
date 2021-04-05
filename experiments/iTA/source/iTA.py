from os.path import isfile
import os
import sys
import time

import re
import numpy as np
import tensorflow as tf

from docqa.data_processing.document_splitter import MergeParagraphs, TopTfIdf, ShallowOpenWebRanker, PreserveParagraphs, DocumentSplitter
from docqa.data_processing.qa_training_data import ParagraphAndQuestion, ParagraphAndQuestionSpec
from docqa.data_processing.text_utils import NltkAndPunctTokenizer, NltkPlusStopWords
from docqa.doc_qa_models import ParagraphQuestionModel
from docqa.model_dir import ModelDir
from docqa.utils import flatten_iterable

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from eli5_utils import qa_s2s_generate
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction

class Loading_Model():
    def __init__(self):
        #bart_tokenizer = AutoTokenizer.from_pretrained('yjernite/bart_eli5')
        #bart_model = AutoModelForSeq2SeqLM.from_pretrained('yjernite/bart_eli5').to('cuda:1')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        args_model = "/home/bdlabucdenver/document_qa/docqa/models/triviaqa-unfiltered-shared-norm/"
        # Load the model
        self.model_dir = ModelDir(args_model)
        self.model = self.model_dir.get_model()
        if not isinstance(self.model, ParagraphQuestionModel):
            raise ValueError("This script is built to work for ParagraphQuestionModel models only")

        # Read the documents
        args_docs = ["/home/bdlabucdenver/data/dataset.txt"]
        documents = []
        for doc in args_docs:
            if not isfile(doc):
                raise ValueError(doc + " does not exist")
            with open(doc, "r") as f:
                documents.append(f.read())
        print("Loaded %d documents" % len(documents))
        
        print("\nPrinting documents before tokenize_paragraph:\n")
        for doc in documents:
            print(doc)
            print("\n")
        
        # Split documents into lists of paragraphs
        documents = [re.split("\s*\n\s*", doc) for doc in documents]
        
        print("\nPrinting documents before tokenize_paragraph:\n")
        for doc in documents:
            print(doc)
            print("\n")
            

        self.tokenizer = NltkAndPunctTokenizer()
        documents = [[self.tokenizer.tokenize_paragraph(p) for p in doc] for doc in documents]
        
        print("\nPrinting documents after tokenize_paragraph:\n")
        for doc in documents:
            print(doc)
            print("\n")
            
        splitter = MergeParagraphs(400)
        self.documents = [splitter.split(doc) for doc in documents]
        
        print("\nPrinting documents after splitter.split:\n")
        for doc in documents:
            print(doc)
            print("\n")
        
        print("\nlength of documents:\n")
        print(len(self.documents))
        
        #q = input("Enter the Question: ")
        # Tokenize the input, the models expects data to be tokenized using `NltkAndPunctTokenizer`
        # Note the model expects case-sensitive input
        
        # Now list of document->paragraph->sentence->word
        

        # Now group the document into paragraphs, this returns `ExtractedParagraph` objects
        # that additionally remember the start/end token of the paragraph within the source document
        
        # splitter = PreserveParagraphs() # Uncomment to use the natural paragraph grouping
        self.bart_tokenizer = AutoTokenizer.from_pretrained('yjernite/bart_eli5')
        self.bart_model = AutoModelForSeq2SeqLM.from_pretrained('yjernite/bart_eli5').to('cuda:0')

    def get_response(self, q):
        start = time.time()
        question = self.tokenizer.tokenize_paragraph_flat(q)  # List of words
        # Now select the top paragraphs using a `ParagraphFilter`
        if len(self.documents) == 1:
            # Use TF-IDF to select top paragraphs from the document
            selector = TopTfIdf(NltkPlusStopWords(True), n_to_select=5)
            context = selector.prune(question, self.documents[0])
        else:
            # Use a linear classifier to select top paragraphs among all the documents
            selector = ShallowOpenWebRanker(n_to_select=10)
            context = selector.prune(question, flatten_iterable(self.documents))

        
        paras = [" ".join(flatten_iterable(x.text)) for x in context]

        print("\nSelected %d paragraphs" % len(context))
        if self.model.preprocessor is not None:
            # Models are allowed to define an additional pre-processing step
            # This will turn the `ExtractedParagraph` objects back into simple lists of tokens
            context = [self.model.preprocessor.encode_text(question, x) for x in context]
        else:
            # Otherwise just use flattened text
            context = [flatten_iterable(x.text) for x in context]

        #print("ACTUALL CONTEXT:\n" + str(context))
        print("\n ---Setting up model---\n")
        # Tell the model the batch size (can be None) and vocab to expect, This will load the
        # needed word vectors and fix the batch size to use when building the graph / encoding the input
        voc = set(question)
        for txt in context:
            voc.update(txt)
        self.model.set_input_spec(ParagraphAndQuestionSpec(batch_size=len(context)), voc)

        # Now we build the actual tensorflow graph, `best_span` and `conf` are
        # tensors holding the predicted span (inclusive) and confidence scores for each
        # element in the input batch, confidence scores being the pre-softmax logit for the span
        #print("Build tf graph")
        config=tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config = config)
        # We need to use sess.as_default when working with the cuNND stuff, since we need an active
        # session to figure out the # of parameters needed for each layer. The cpu-compatible models don't need this.
        with sess.as_default():
            # 8 means to limit the span to size 8 or less
            best_spans, conf = self.model.get_prediction().get_best_span(8)

        # Loads the saved weights
        self.model_dir.restore_checkpoint(sess)

        # Now the model is ready to run
        # The model takes input in the form of `ContextAndQuestion` objects, for example:
        data = [ParagraphAndQuestion(x, question, None, "user-question%d"%i)
                for i, x in enumerate(context)]

        # The model is run in two steps, first it "encodes" a batch of paragraph/context pairs
        # into numpy arrays, then we use `sess` to run the actual model get the predictions
        encoded = self.model.encode(data, is_train=False)  # batch of `ContextAndQuestion` -> feed_dict
        best_spans, conf = sess.run([best_spans, conf], feed_dict=encoded)  # feed_dict -> predictions

        best_para = np.argmax(conf)  # We get output for each paragraph, select the most-confident one to print

        best_paras = np.argsort(conf)
        end = time.time()
        print(end- start)

        #top_para = q + " --T-- " + paras[best_paras[4]] + " <D> " + paras[best_paras[3]] + " <D> " + paras[best_paras[2]]
        
        #top_para = q + " --T-- " + paras[best_para]
        top_para = "question: " + q + " context: " + paras[best_paras[4]] + paras[best_paras[3]]

        print("\nAnswer by TriviQA:\n")
        #print("Paragraph Order:" + str(best_paras))
        print("Best Paragraph: " + str(best_para))
        #print("Best span: " + str(best_spans[best_para]))
        print("Answer text: " + " ".join(context[best_para][best_spans[best_para][0]:best_spans[best_para][1]+1]))
        #print("Confidence: " + str(conf[best_para]))
        top_file = open("/home/bdlabucdenver/Top_para.txt",'w')
        top_file.write(top_para)
        top_file.close()
        answer = qa_s2s_generate(top_para, self.bart_model, self.bart_tokenizer,num_answers=1,num_beams=8, min_len=96, max_len=256, max_input_length=1024, device='cuda:0')[0]
        tf.get_variable_scope().reuse_variables()
        return answer

    



