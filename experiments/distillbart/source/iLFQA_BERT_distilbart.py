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

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, BartTokenizer, BartForConditionalGeneration, BartConfig, pipeline
from eli5_utils import qa_s2s_generate
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction

class Loading_Model():
    def __init__(self):
        
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
        
        # load the zero-shot classifier from hugging face
        classifier_name = "zero-shot-classification"
        self.zshot_classifier = pipeline(classifier_name, device=0)
        
        # Create dictionary to classify textbooks
        self.topics = {
        
            "data": ["dataset.txt"],
            "data science": ["dataset.txt"],
            "statistics": ["Classical-Sociology.txt", "dataset.txt"],
            "behavior": ["Human-Behavior.txt", "Classical-Sociology.txt"],
            "energy": ["Direct_Energy_Conversion.txt"],
            "history": ["History_of_International_Relations.txt", "Classical-Sociology.txt"],
            "politics": ["Classical-Sociology.txt", "History_of_International_Relations.txt"],
            "international relations":["History_of_International_Relations.txt"],
            "sociology": ["Classical-Sociology.txt"],
            "social science": ["Classical-Sociology.txt"],
            "people": ["Human-Behavior.txt", "Classical-Sociology.txt", "History_of_International_Relations.txt"]
        
        }
        
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
        self.text_books = text_books

        #q = input("Enter the Question: ")
        # Tokenize the input, the models expects data to be tokenized using `NltkAndPunctTokenizer`
        # Note the model expects case-sensitive input
        
        # Now list of document->paragraph->sentence->word
        

        # Now group the document into paragraphs, this returns `ExtractedParagraph` objects
        # that additionally remember the start/end token of the paragraph within the source document
        
        # splitter = PreserveParagraphs() # Uncomment to use the natural paragraph grouping
        model_name = "sshleifer/distilbart-cnn-6-6"
        self.bart_tokenizer = BartTokenizer.from_pretrained(model_name)
        self.bart_model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # Load the BERT model we use to determine paragraph confidence scores
        model_name = "deepset/roberta-base-squad2"
        self.nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=0)
    
    def two_answer_best_context(self, q):

        start = time.time()
        
        # Classify the question
        zero_shot_time = time.time()
        # Use the classifier to classify the question.
        candidate_labels = list(self.topics.keys())
        zero_shot_result =  self.zshot_classifier(q, candidate_labels)
        print("\nzero_shot_results:\n{}".format(zero_shot_result))
        
        # Find the index with highest score
        score_index = 0
        best_score = {"topic": '', "score": -1}
        for score in zero_shot_result['scores']:
            if score > best_score["score"]:
                best_score["score"] = score
                best_score["topic"] = zero_shot_result['labels'][score_index]
            score_index += 1
        print("\nbest_score:\n{}".format(best_score))
        
        # Make a list of only the desired documents
        topic_texts = self.topics[best_score["topic"]]
        print("\ntopic_texts:\n{}\n".format(topic_texts))
        topic_texts_index = [self.text_books.index(text) for text in topic_texts]
        print("\ntopic_texts_index:\n{}\n".format(topic_texts_index))
        print("\nself.text_books:\n{}\n".format(self.text_books))
        selected_texts = []
        for index in topic_texts_index:
            index_number = index
            print("\nindex_number:{}\n".format(index_number))
            selected_texts.append(self.documents[index_number])
        # print("\nlength of selected_texts:\n{}\n".format(len(selected_texts))
        
        end_zero_shot_time = time.time()
        total_zero_shot_time = end_zero_shot_time - zero_shot_time
        print("\ntotal_zero_shot_time:{}\n".format(total_zero_shot_time))
        
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
        tf_idf_time = endTimeForTFIDF - timeForTFIDF
        print('Total time for TFIDF: {}'.format(tf_idf_time))
       
        question_answer_dict_list = list()

        for paragraph in paras:
            question_answer_dict_list.append({'question': q, 'context': paragraph})

        scoreTime = time.time()
        for question in question_answer_dict_list:
            response = self.nlp(question)
            question['score'] = response['score']
            question['answer'] = response['answer']
        endScoreTime = time.time()
        confidence_score_time = endScoreTime - scoreTime
        print('Total time for scoring: {}'.format(confidence_score_time))
        
        # We want to get the list in descending order from best confidence score to worst.
        question_answer_dict_list_sorted = sorted(question_answer_dict_list, key = lambda i: i['score'], reverse=True)
        
        answer = "No answer yet"
        
        # We will pass the top two answers along with the highest scored context to BART
        top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[1]['answer'] + ' ' + question_answer_dict_list_sorted[2]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context']
        context = question_answer_dict_list_sorted[1]['answer'] + ' ' + question_answer_dict_list_sorted[2]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context']
        
        
        # Change top_para into a dict for easier access
        top_para = {
            "question": q,
            "context": question_answer_dict_list_sorted[0]['answer'] + ' ' + question_answer_dict_list_sorted[1]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context']
        }
        
        print('\ntop_para: \n{}\n'.format(top_para))

        answerTime = time.time()
        # Generate the inputs for the summary to use
        inputs = self.bart_tokenizer([top_para['context']], max_length=1024, return_tensors='pt')
        # Generate summary itself min_length and max_length specify summary length
        summary_ids = self.bart_model.generate(inputs['input_ids'], num_beams=4, min_length = 30, max_length=100, early_stopping=True)
        answer = ''
        for g in summary_ids:
            answer = answer + self.bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(answer)
        tf.get_variable_scope().reuse_variables()
        endAnswerTime = time.time()
        total_answer_time = endAnswerTime - answerTime
        print('\nTotal time to generate answer: {}\n'.format(total_answer_time))
        
        timeDict = {"tf_idf": tf_idf_time, "confidence_scores": confidence_score_time, "answer": total_answer_time, "zero_shot_time": total_zero_shot_time}
        
        return answer, timeDict, context

    # end def two_answer_best_context(self, q)
    
    def four_best_context_concat(self, q):

        start = time.time()
        
        # Classify the question
        zero_shot_time = time.time()
        # Use the classifier to classify the question.
        candidate_labels = list(self.topics.keys())
        zero_shot_result =  self.zshot_classifier(q, candidate_labels)
        print("\nzero_shot_results:\n{}".format(zero_shot_result))
        
        # Find the index with highest score
        score_index = 0
        best_score = {"topic": '', "score": -1}
        for score in zero_shot_result['scores']:
            if score > best_score["score"]:
                best_score["score"] = score
                best_score["topic"] = zero_shot_result['labels'][score_index]
            score_index += 1
        print("\nbest_score:\n{}".format(best_score))
        
        # Make a list of only the desired documents
        topic_texts = self.topics[best_score["topic"]]
        print("\ntopic_texts:\n{}\n".format(topic_texts))
        topic_texts_index = [self.text_books.index(text) for text in topic_texts]
        print("\ntopic_texts_index:\n{}\n".format(topic_texts_index))
        print("\nself.text_books:\n{}\n".format(self.text_books))
        selected_texts = []
        for index in topic_texts_index:
            index_number = index
            print("\nindex_number:{}\n".format(index_number))
            selected_texts.append(self.documents[index_number])
        # print("\nlength of selected_texts:\n{}\n".format(len(selected_texts))
        
        end_zero_shot_time = time.time()
        total_zero_shot_time = end_zero_shot_time - zero_shot_time
        print("\ntotal_zero_shot_time:{}\n".format(total_zero_shot_time))
        
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
        tf_idf_time = endTimeForTFIDF - timeForTFIDF
        print('Total time for TFIDF: {}'.format(tf_idf_time))
       
        question_answer_dict_list = list()

        for paragraph in paras:
            question_answer_dict_list.append({'question': q, 'context': paragraph})

        scoreTime = time.time()
        for question in question_answer_dict_list:
            response = self.nlp(question)
            question['score'] = response['score']
            question['answer'] = response['answer']
        endScoreTime = time.time()
        confidence_score_time = endScoreTime - scoreTime
        print('Total time for scoring: {}'.format(confidence_score_time))
        
        # We want to get the list in descending order from best confidence score to worst.
        question_answer_dict_list_sorted = sorted(question_answer_dict_list, key = lambda i: i['score'], reverse=True)
        
        answer = "No answer yet"
        
        # We will pass the top two answers along with the highest scored context to BART
        top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[0]['context'] + ' ' + question_answer_dict_list_sorted[1]['context'] + ' ' + question_answer_dict_list_sorted[2]['context'] + ' ' + question_answer_dict_list_sorted[3]['context']
        context = question_answer_dict_list_sorted[0]['context'] + ' ' + question_answer_dict_list_sorted[1]['context'] + ' ' + question_answer_dict_list_sorted[2]['context'] + ' ' + question_answer_dict_list_sorted[3]['context']
        
        
        # Change top_para into a dict for easier access
        top_para = {
            "question": q,
            "context": question_answer_dict_list_sorted[0]['context'] + ' ' + question_answer_dict_list_sorted[1]['context'] + ' ' + question_answer_dict_list_sorted[2]['context'] + ' ' + question_answer_dict_list_sorted[3]['context']
        }
        
        print('\ntop_para: \n{}\n'.format(top_para))

        answerTime = time.time()
        # Generate the inputs for the summary to use
        inputs = self.bart_tokenizer([top_para['context']], max_length=1024, return_tensors='pt')
        # Generate summary itself min_length and max_length specify summary length
        summary_ids = self.bart_model.generate(inputs['input_ids'], num_beams=4, min_length = 30, max_length=100, early_stopping=True)
        answer = ''
        for g in summary_ids:
            answer = answer + self.bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(answer)
        tf.get_variable_scope().reuse_variables()
        endAnswerTime = time.time()
        total_answer_time = endAnswerTime - answerTime
        print('\nTotal time to generate answer: {}\n'.format(total_answer_time))
        
        timeDict = {"tf_idf": tf_idf_time, "confidence_scores": confidence_score_time, "answer": total_answer_time, "zero_shot_time": total_zero_shot_time}
        
        return answer, timeDict, context

    # end def four_best_context_concat(self, q)
    
    def five_answer_best_context_concat(self, q):

        start = time.time()
        
        # Classify the question
        zero_shot_time = time.time()
        # Use the classifier to classify the question.
        candidate_labels = list(self.topics.keys())
        zero_shot_result =  self.zshot_classifier(q, candidate_labels)
        print("\nzero_shot_results:\n{}".format(zero_shot_result))
        
        # Find the index with highest score
        score_index = 0
        best_score = {"topic": '', "score": -1}
        for score in zero_shot_result['scores']:
            if score > best_score["score"]:
                best_score["score"] = score
                best_score["topic"] = zero_shot_result['labels'][score_index]
            score_index += 1
        print("\nbest_score:\n{}".format(best_score))
        
        # Make a list of only the desired documents
        topic_texts = self.topics[best_score["topic"]]
        print("\ntopic_texts:\n{}\n".format(topic_texts))
        topic_texts_index = [self.text_books.index(text) for text in topic_texts]
        print("\ntopic_texts_index:\n{}\n".format(topic_texts_index))
        print("\nself.text_books:\n{}\n".format(self.text_books))
        selected_texts = []
        for index in topic_texts_index:
            index_number = index
            print("\nindex_number:{}\n".format(index_number))
            selected_texts.append(self.documents[index_number])
        # print("\nlength of selected_texts:\n{}\n".format(len(selected_texts))
        
        end_zero_shot_time = time.time()
        total_zero_shot_time = end_zero_shot_time - zero_shot_time
        print("\ntotal_zero_shot_time:{}\n".format(total_zero_shot_time))
        
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
        tf_idf_time = endTimeForTFIDF - timeForTFIDF
        print('Total time for TFIDF: {}'.format(tf_idf_time))
       
        question_answer_dict_list = list()

        for paragraph in paras:
            question_answer_dict_list.append({'question': q, 'context': paragraph})

        scoreTime = time.time()
        for question in question_answer_dict_list:
            response = self.nlp(question)
            question['score'] = response['score']
            question['answer'] = response['answer']
        endScoreTime = time.time()
        confidence_score_time = endScoreTime - scoreTime
        print('Total time for scoring: {}'.format(confidence_score_time))
        
        # We want to get the list in descending order from best confidence score to worst.
        question_answer_dict_list_sorted = sorted(question_answer_dict_list, key = lambda i: i['score'], reverse=True)
        
        answer = "No answer yet"
        
        # We will pass the top two answers along with the highest scored context to BART
        top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[1]['answer'] + ' ' + question_answer_dict_list_sorted[2]['answer'] + ' ' + question_answer_dict_list_sorted[3]['answer'] + ' ' + question_answer_dict_list_sorted[4]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context']
        context = question_answer_dict_list_sorted[1]['answer'] + ' ' + question_answer_dict_list_sorted[2]['answer'] + ' ' + question_answer_dict_list_sorted[3]['answer'] + ' ' + question_answer_dict_list_sorted[4]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context']
        
        
        # Change top_para into a dict for easier access
        top_para = {
            "question": q,
            "context": question_answer_dict_list_sorted[1]['answer'] + ' ' + question_answer_dict_list_sorted[2]['answer'] + ' ' + question_answer_dict_list_sorted[3]['answer'] + ' ' + question_answer_dict_list_sorted[4]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context']
        }
        
        print('\ntop_para: \n{}\n'.format(top_para))

        answerTime = time.time()
        # Generate the inputs for the summary to use
        inputs = self.bart_tokenizer([top_para['context']], max_length=1024, return_tensors='pt')
        # Generate summary itself min_length and max_length specify summary length
        summary_ids = self.bart_model.generate(inputs['input_ids'], num_beams=4, min_length = 30, max_length=100, early_stopping=True)
        answer = ''
        for g in summary_ids:
            answer = answer + self.bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(answer)
        tf.get_variable_scope().reuse_variables()
        endAnswerTime = time.time()
        total_answer_time = endAnswerTime - answerTime
        print('\nTotal time to generate answer: {}\n'.format(total_answer_time))
        
        timeDict = {"tf_idf": tf_idf_time, "confidence_scores": confidence_score_time, "answer": total_answer_time, "zero_shot_time": total_zero_shot_time}
        
        return answer, timeDict, context

    # end def five_answer_best_context_concat(self, q)
    
    def two_answer_two_context_concat(self, q):

        start = time.time()
        
        # Classify the question
        zero_shot_time = time.time()
        # Use the classifier to classify the question.
        candidate_labels = list(self.topics.keys())
        zero_shot_result =  self.zshot_classifier(q, candidate_labels)
        print("\nzero_shot_results:\n{}".format(zero_shot_result))
        
        # Find the index with highest score
        score_index = 0
        best_score = {"topic": '', "score": -1}
        for score in zero_shot_result['scores']:
            if score > best_score["score"]:
                best_score["score"] = score
                best_score["topic"] = zero_shot_result['labels'][score_index]
            score_index += 1
        print("\nbest_score:\n{}".format(best_score))
        
        # Make a list of only the desired documents
        topic_texts = self.topics[best_score["topic"]]
        print("\ntopic_texts:\n{}\n".format(topic_texts))
        topic_texts_index = [self.text_books.index(text) for text in topic_texts]
        print("\ntopic_texts_index:\n{}\n".format(topic_texts_index))
        print("\nself.text_books:\n{}\n".format(self.text_books))
        selected_texts = []
        for index in topic_texts_index:
            index_number = index
            print("\nindex_number:{}\n".format(index_number))
            selected_texts.append(self.documents[index_number])
        # print("\nlength of selected_texts:\n{}\n".format(len(selected_texts))
        
        end_zero_shot_time = time.time()
        total_zero_shot_time = end_zero_shot_time - zero_shot_time
        print("\ntotal_zero_shot_time:{}\n".format(total_zero_shot_time))
        
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
        tf_idf_time = endTimeForTFIDF - timeForTFIDF
        print('Total time for TFIDF: {}'.format(tf_idf_time))
       
        question_answer_dict_list = list()

        for paragraph in paras:
            question_answer_dict_list.append({'question': q, 'context': paragraph})

        scoreTime = time.time()
        for question in question_answer_dict_list:
            response = self.nlp(question)
            question['score'] = response['score']
            question['answer'] = response['answer']
        endScoreTime = time.time()
        confidence_score_time = endScoreTime - scoreTime
        print('Total time for scoring: {}'.format(confidence_score_time))
        
        # We want to get the list in descending order from best confidence score to worst.
        question_answer_dict_list_sorted = sorted(question_answer_dict_list, key = lambda i: i['score'], reverse=True)
        answer = "No answer yet"
        
        # We will pass the top two answers along with the highest scored context to BART
        top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[2]['answer'] + ' ' + question_answer_dict_list_sorted[3]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context'] + ' ' + question_answer_dict_list_sorted[1]['context']
        context = question_answer_dict_list_sorted[2]['answer'] + ' ' + question_answer_dict_list_sorted[3]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context'] + ' ' + question_answer_dict_list_sorted[1]['context']
        
        
        # Change top_para into a dict for easier access
        top_para = {
            "question": q,
            "context": question_answer_dict_list_sorted[2]['answer'] + ' ' + question_answer_dict_list_sorted[3]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context'] + ' ' + question_answer_dict_list_sorted[1]['context']
        }
        
        print('\ntop_para: \n{}\n'.format(top_para))

        answerTime = time.time()
        # Generate the inputs for the summary to use
        inputs = self.bart_tokenizer([top_para['context']], max_length=1024, return_tensors='pt')
        # Generate summary itself min_length and max_length specify summary length
        summary_ids = self.bart_model.generate(inputs['input_ids'], num_beams=4, min_length = 30, max_length=100, early_stopping=True)
        answer = ''
        for g in summary_ids:
            answer = answer + self.bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(answer)
        tf.get_variable_scope().reuse_variables()
        endAnswerTime = time.time()
        total_answer_time = endAnswerTime - answerTime
        print('\nTotal time to generate answer: {}\n'.format(total_answer_time))
        
        timeDict = {"tf_idf": tf_idf_time, "confidence_scores": confidence_score_time, "answer": total_answer_time, "zero_shot_time": total_zero_shot_time}
        
        return answer, timeDict, context

    # end def two_answer_two_context_concat(self, q)
