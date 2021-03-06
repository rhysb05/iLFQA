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
from pipelines import pipeline as pipelines_pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelWithLMHead,AutoModelForQuestionAnswering, pipeline
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
        doc_file_path = "/home/bdlabucdenver/data/texts/"
        text_books = ["Classical-Sociology.txt","dataset.txt","Direct_Energy_Conversion.txt","History_of_International_Relations.txt","Human-Behavior.txt"]
        args_docs = []
        for text_book in text_books:
            text_book = doc_file_path + text_book
            args_docs.append(text_book)
        documents = []
        for doc in args_docs:
            if not isfile(doc):
                raise ValueError(doc + " does not exist")
            with open(doc, "r") as f:
                print(doc)
                documents.append(f.read())
        print("Loaded %d documents" % len(documents))

        # Split documents into lists of paragraphs
        documents = [re.split("\s*\n\s*", doc) for doc in documents]

        self.tokenizer = NltkAndPunctTokenizer()
        documents = [[self.tokenizer.tokenize_paragraph(p) for p in doc] for doc in documents]
        splitter = MergeParagraphs(400)
        self.documents = [splitter.split(doc) for doc in documents]

        #q = input("Enter the Question: ")
        # Tokenize the input, the models expects data to be tokenized using `NltkAndPunctTokenizer`
        # Note the model expects case-sensitive input
        
        # Now list of document->paragraph->sentence->word
        

        # Now group the document into paragraphs, this returns `ExtractedParagraph` objects
        # that additionally remember the start/end token of the paragraph within the source document
        
        
        # Load the BERT model we use to determine paragraph confidence scores
        model_name = "deepset/roberta-base-squad2"
        self.nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
        
        # Load the T5 model for summarization
        # self.T5_summ_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news", model_max_length=1000)
        self.T5_summ_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news", model_max_length=2048)
        self.T5_summ_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
        
    
    
    def summarize(self, text, max_length):
        
        input_ids = self.T5_summ_tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
        
        # generated_ids = self.T5_summ_model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=False)
        generated_ids = self.T5_summ_model.generate(input_ids=input_ids, num_beams=2, max_length=max_length,  repetition_penalty=2.5, length_penalty=0.1, early_stopping=False)

        preds = [self.T5_summ_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        
        return preds[0]

    
    def get_response_BERT_T5_summ(self, q):

        start = time.time()
        timeForTFIDF = time.time()
        question = self.tokenizer.tokenize_paragraph_flat(q)  # List of words
        # Now select the top paragraphs using a `ParagraphFilter`
        if len(self.documents) == 1:
            # Use TF-IDF to select top paragraphs from the document
            selector = TopTfIdf(NltkPlusStopWords(True), n_to_select=5)
            context = selector.prune(question, self.documents[0])
        else:
            # Use a linear classifier to select top paragraphs among all the documents
            selector = ShallowOpenWebRanker(n_to_select=5)
            context = selector.prune(question, flatten_iterable(self.documents))

        
        paras = [" ".join(flatten_iterable(x.text)) for x in context]
        endTimeForTFIDF = time.time()
        print('Total time for TFIDF: {}'.format(endTimeForTFIDF - timeForTFIDF))
       
        question_answer_dict_list = list()

        for paragraph in paras:
            question_answer_dict_list.append({'question': q, 'context': paragraph})

        scoreTime = time.time()
        for question in question_answer_dict_list:
            response = self.nlp(question)
            question['score'] = response['score']
            question['answer'] = response['answer']
        endScoreTime = time.time()
        print('Total time for scoring: {}'.format(endScoreTime - scoreTime))
        
        # We want to get the list in descending order from best confidence score to worst.
        question_answer_dict_list_sorted = sorted(question_answer_dict_list, key = lambda i: i['score'], reverse=True)
        
        answer = "No answer yet"
        
        # We will pass the top two answers along with the highest scored context to BART
        # top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[1]['answer'] + ' ' + question_answer_dict_list_sorted[2]['answer']  + ' ' + question_answer_dict_list_sorted[3]['answer'] + ' ' + question_answer_dict_list_sorted[4]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context']
        top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[0]['answer'] + ' ' + question_answer_dict_list_sorted[1]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context']
        print('\n{} {}\n'.format(question_answer_dict_list_sorted[0]['answer'], question_answer_dict_list_sorted[1]['answer']))
        print('\n{}\n'.format(question_answer_dict_list_sorted[4]['answer']))
        print('\n{} {}\n'.format(question_answer_dict_list_sorted[2]['answer'], question_answer_dict_list_sorted[3]['answer']))
        print('\n {} \n'.format(top_para))

        # print("\nAnswer by TriviQA:\n")
        # #print("Paragraph Order:" + str(best_paras))
        # print("Best Paragraph: " + str(best_para))
        # #print("Best span: " + str(best_spans[best_para]))
        # print("Answer text: " + " ".join(context[best_para][best_spans[best_para][0]:best_spans[best_para][1]+1]))
        # #print("Confidence: " + str(conf[best_para]))
        top_file = open("/home/bdlabucdenver/Top_para.txt",'w')
        top_file.write(top_para)
        top_file.close()

        answerTime = time.time()
        # Using T5 model to generate answer
        answer = self.summarize(top_para, 80)
        # tf.get_variable_scope().reuse_variables()
        endAnswerTime = time.time()
        tf.get_variable_scope().reuse_variables()
        print('\nTotal time to generate answer: {}\n'.format(endAnswerTime - answerTime))

        return answer

    # end def get_response_BERT_answer_concat(self, q)
    
    def T5_summ_2_concat_para(self, q):
    
        start = time.time()
        timeForTFIDF = time.time()
        question = self.tokenizer.tokenize_paragraph_flat(q)  # List of words
        # Now select the top paragraphs using a `ParagraphFilter`
        if len(self.documents) == 1:
            # Use TF-IDF to select top paragraphs from the document
            selector = TopTfIdf(NltkPlusStopWords(True), n_to_select=5)
            context = selector.prune(question, self.documents[0])
        else:
            # Use a linear classifier to select top paragraphs among all the documents
            selector = ShallowOpenWebRanker(n_to_select=5)
            context = selector.prune(question, flatten_iterable(self.documents))

        
        paras = [" ".join(flatten_iterable(x.text)) for x in context]
        endTimeForTFIDF = time.time()
        print('Total time for TFIDF: {}'.format(endTimeForTFIDF - timeForTFIDF))
       
        question_answer_dict_list = list()

        for paragraph in paras:
            question_answer_dict_list.append({'question': q, 'context': paragraph})

        scoreTime = time.time()
        for question in question_answer_dict_list:
            response = self.nlp(question)
            question['score'] = response['score']
            question['answer'] = response['answer']
        endScoreTime = time.time()
        print('Total time for scoring: {}'.format(endScoreTime - scoreTime))
        
        # We want to get the list in descending order from best confidence score to worst.
        question_answer_dict_list_sorted = sorted(question_answer_dict_list, key = lambda i: i['score'], reverse=True)
        
        answer = "No answer yet"
        
        # We will pass the top two answers along with the highest scored context to BART
        # top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[1]['answer'] + ' ' + question_answer_dict_list_sorted[2]['answer']  + ' ' + question_answer_dict_list_sorted[3]['answer'] + ' ' + question_answer_dict_list_sorted[4]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context']
        top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[0]['context'] + ' ' + question_answer_dict_list_sorted[1]['context']
       
        print('\ntop_para:\n{}\n'.format(top_para)) 
        
        # print("\nAnswer by TriviQA:\n")
        # #print("Paragraph Order:" + str(best_paras))
        # print("Best Paragraph: " + str(best_para))
        # #print("Best span: " + str(best_spans[best_para]))
        # print("Answer text: " + " ".join(context[best_para][best_spans[best_para][0]:best_spans[best_para][1]+1]))
        # #print("Confidence: " + str(conf[best_para]))
        top_file = open("/home/bdlabucdenver/Top_para.txt",'w')
        top_file.write(top_para)
        top_file.close()

        answerTime = time.time()
        # Using T5 model to generate answer
        answer = self.summarize(top_para, 150)
        # tf.get_variable_scope().reuse_variables()
        endAnswerTime = time.time()
        tf.get_variable_scope().reuse_variables()
        print('\nTotal time to generate answer: {}\n'.format(endAnswerTime - answerTime))

        return answer

    # end def T5_summ_2_concat_para(self, q)
    
    def get_response_BERT_T5_summ(self, q):

        start = time.time()
        timeForTFIDF = time.time()
        question = self.tokenizer.tokenize_paragraph_flat(q)  # List of words
        # Now select the top paragraphs using a `ParagraphFilter`
        if len(self.documents) == 1:
            # Use TF-IDF to select top paragraphs from the document
            selector = TopTfIdf(NltkPlusStopWords(True), n_to_select=5)
            context = selector.prune(question, self.documents[0])
        else:
            # Use a linear classifier to select top paragraphs among all the documents
            selector = ShallowOpenWebRanker(n_to_select=5)
            context = selector.prune(question, flatten_iterable(self.documents))

        
        paras = [" ".join(flatten_iterable(x.text)) for x in context]
        endTimeForTFIDF = time.time()
        print('Total time for TFIDF: {}'.format(endTimeForTFIDF - timeForTFIDF))
       
        question_answer_dict_list = list()

        for paragraph in paras:
            question_answer_dict_list.append({'question': q, 'context': paragraph})

        scoreTime = time.time()
        for question in question_answer_dict_list:
            response = self.nlp(question)
            question['score'] = response['score']
            question['answer'] = response['answer']
        endScoreTime = time.time()
        print('Total time for scoring: {}'.format(endScoreTime - scoreTime))
        
        # We want to get the list in descending order from best confidence score to worst.
        question_answer_dict_list_sorted = sorted(question_answer_dict_list, key = lambda i: i['score'], reverse=True)
        
        answer = "No answer yet"
        
        # We will pass the top two answers along with the highest scored context to BART
        # top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[1]['answer'] + ' ' + question_answer_dict_list_sorted[2]['answer']  + ' ' + question_answer_dict_list_sorted[3]['answer'] + ' ' + question_answer_dict_list_sorted[4]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context']
        top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[0]['answer'] + ' ' + question_answer_dict_list_sorted[1]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context']
        print('\n{} {}\n'.format(question_answer_dict_list_sorted[0]['answer'], question_answer_dict_list_sorted[1]['answer']))
        print('\n{}\n'.format(question_answer_dict_list_sorted[4]['answer']))
        print('\n{} {}\n'.format(question_answer_dict_list_sorted[2]['answer'], question_answer_dict_list_sorted[3]['answer']))
        print('\n {} \n'.format(top_para))

        # print("\nAnswer by TriviQA:\n")
        # #print("Paragraph Order:" + str(best_paras))
        # print("Best Paragraph: " + str(best_para))
        # #print("Best span: " + str(best_spans[best_para]))
        # print("Answer text: " + " ".join(context[best_para][best_spans[best_para][0]:best_spans[best_para][1]+1]))
        # #print("Confidence: " + str(conf[best_para]))
        top_file = open("/home/bdlabucdenver/Top_para.txt",'w')
        top_file.write(top_para)
        top_file.close()

        answerTime = time.time()
        # Using T5 model to generate answer
        answer = self.summarize(top_para, 80)
        # tf.get_variable_scope().reuse_variables()
        endAnswerTime = time.time()
        tf.get_variable_scope().reuse_variables()
        print('\nTotal time to generate answer: {}\n'.format(endAnswerTime - answerTime))

        return answer

    # end def get_response_BERT_answer_concat(self, q)
    
    def T5_summ_4_concat_para(self, q):
    
        start = time.time()
        timeForTFIDF = time.time()
        question = self.tokenizer.tokenize_paragraph_flat(q)  # List of words
        # Now select the top paragraphs using a `ParagraphFilter`
        if len(self.documents) == 1:
            # Use TF-IDF to select top paragraphs from the document
            selector = TopTfIdf(NltkPlusStopWords(True), n_to_select=5)
            context = selector.prune(question, self.documents[0])
        else:
            # Use a linear classifier to select top paragraphs among all the documents
            selector = ShallowOpenWebRanker(n_to_select=5)
            context = selector.prune(question, flatten_iterable(self.documents))

        
        paras = [" ".join(flatten_iterable(x.text)) for x in context]
        endTimeForTFIDF = time.time()
        print('Total time for TFIDF: {}'.format(endTimeForTFIDF - timeForTFIDF))
       
        question_answer_dict_list = list()

        for paragraph in paras:
            question_answer_dict_list.append({'question': q, 'context': paragraph})

        scoreTime = time.time()
        for question in question_answer_dict_list:
            response = self.nlp(question)
            question['score'] = response['score']
            question['answer'] = response['answer']
        endScoreTime = time.time()
        print('Total time for scoring: {}'.format(endScoreTime - scoreTime))
        
        # We want to get the list in descending order from best confidence score to worst.
        question_answer_dict_list_sorted = sorted(question_answer_dict_list, key = lambda i: i['score'], reverse=True)
        
        answer = "No answer yet"
        
        # We will pass the top two answers along with the highest scored context to BART
        # top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[1]['answer'] + ' ' + question_answer_dict_list_sorted[2]['answer']  + ' ' + question_answer_dict_list_sorted[3]['answer'] + ' ' + question_answer_dict_list_sorted[4]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context']
        top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[0]['context'] + ' ' + question_answer_dict_list_sorted[1]['context'] + ' ' + question_answer_dict_list_sorted[2]['context'] + ' ' + question_answer_dict_list_sorted[3]['context']
       
        print('\ntop_para:\n{}\n'.format(top_para)) 
        
        # print("\nAnswer by TriviQA:\n")
        # #print("Paragraph Order:" + str(best_paras))
        # print("Best Paragraph: " + str(best_para))
        # #print("Best span: " + str(best_spans[best_para]))
        # print("Answer text: " + " ".join(context[best_para][best_spans[best_para][0]:best_spans[best_para][1]+1]))
        # #print("Confidence: " + str(conf[best_para]))
        top_file = open("/home/bdlabucdenver/Top_para.txt",'w')
        top_file.write(top_para)
        top_file.close()

        answerTime = time.time()
        # Using T5 model to generate answer
        answer = self.summarize(top_para, 150)
        # tf.get_variable_scope().reuse_variables()
        endAnswerTime = time.time()
        tf.get_variable_scope().reuse_variables()
        print('\nTotal time to generate answer: {}\n'.format(endAnswerTime - answerTime))

        return answer

    # end def T5_summ_2_concat_para(self, q)

