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
        
        # splitter = PreserveParagraphs() # Uncomment to use the natural paragraph grouping
        model_name = "sshleifer/distilbart-cnn-6-6"
        self.bart_tokenizer = BartTokenizer.from_pretrained(model_name)
        self.bart_model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # Load the BERT model we use to determine paragraph confidence scores
        model_name = "deepset/roberta-base-squad2"
        self.nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=0)

    def get_response(self, q):
        
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
            selector = ShallowOpenWebRanker(n_to_select=10)
            context = selector.prune(question, flatten_iterable(self.documents))

        
        paras = [" ".join(flatten_iterable(x.text)) for x in context]
        endTimeForTFIDF = time.time()
        print('Total time for TFIDF: {}'.format(endTimeForTFIDF - timeForTFIDF))

        # timeForModel = time.time()
        # # print(paras)
        # model_name = "deepset/roberta-base-squad2"
        # nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
        # endTimeForModel = time.time()
        # print('Total time to load model: {}'.format(endTimeForModel - timeForModel)) 
       
        # question_answer_dict_list = list()
        
        # for paragraph in paras:
            # question_answer_dict_list.append({'question': q, 'context': paragraph})

        # scoreList = list()
        # scoreTime = time.time()
        # for question in question_answer_dict_list:
            # scoreList.append(nlp(question))
        # endScoreTime = time.time()
        # print('Total time for scoring: {}'.format(endScoreTime - scoreTime))
        # for score in scoreList:
            # print(score)

        # return True



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

    def get_response_BERT(self, q):

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
        response_answer_list = list()

        for paragraph in paras:
            question_answer_dict_list.append({'question': q, 'context': paragraph})

        scoreTime = time.time()
        for question in question_answer_dict_list:
            response = self.nlp(question)
            question['score'] = response['score']
        endScoreTime = time.time()
        print('Total time for scoring: {}'.format(endScoreTime - scoreTime))
        
        answer = "No answer yet"
        
        # We want to get the list in descending order from best confidence score to worst.
        question_answer_dict_list_sorted = sorted(question_answer_dict_list, key = lambda i: i['score'], reverse=True)
        
        # print("\nSorted List:\n")
        # for scoredQuestion in question_answer_dict_list_sorted:
        #     print("\n{}\n".format(scoredQuestion))
        # end for score in scoreList
        

        #top_para = q + " --T-- " + paras[best_para]
        top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[0]['context'] + question_answer_dict_list_sorted[1]['context']
        
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
        answer = qa_s2s_generate(top_para, self.bart_model, self.bart_tokenizer,num_answers=1,num_beams=8, min_len=96, max_len=256, max_input_length=1024, device='cuda:0')[0]
        # tf.get_variable_scope().reuse_variables()
        endAnswerTime = time.time()
        tf.get_variable_scope().reuse_variables()
        print('\nTotal time to generate answer: {}\n'.format(endAnswerTime - answerTime))
        
        return answer

    # end def get_response_BERT(self, q)
    
    def get_response_BERT_answer_concat(self, q):

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
        response_answer_list = list()

        for paragraph in paras:
            question_answer_dict_list.append({'question': q, 'context': paragraph})

        scoreTime = time.time()
        for question in question_answer_dict_list:
            response = self.nlp(question)
            # print(response['answer'])
            # ************************************** Concatenated answers ************************************
            response_answer_list.append(response['answer'])
            # ************************************** Concatenated answers ************************************
        endScoreTime = time.time()
        print('Total time for scoring: {}'.format(endScoreTime - scoreTime))
        
        answer = "No answer yet"
        
        # *********************************************** Concatenated answers *******************************************************************************
        # The following lines are for testing purposes. They concatenate the context of the best spans of the five answers produced by  the BERT
        # confidence scores.
        concatenated_answers = ''
        for responseAnswer in response_answer_list:
            concatenated_answers = concatenated_answers + ' ' + responseAnswer

        print('\nConcatenated answers:\n{}\n'.format(concatenated_answers))
        
        # We will pass the concatenated_answers to BART to generate an answer just for experimentation
        top_para = "question: " + q + " context: " + concatenated_answers
        
        # *********************************************** Concatenated answers end ***************************************************************************

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
        answer = qa_s2s_generate(top_para, self.bart_model, self.bart_tokenizer,num_answers=1,num_beams=8, min_len=96, max_len=256, max_input_length=1024, device='cuda:0')[0]
        # tf.get_variable_scope().reuse_variables()
        endAnswerTime = time.time()
        tf.get_variable_scope().reuse_variables()
        print('\nTotal time to generate answer: {}\n'.format(endAnswerTime - answerTime))


        return answer

    # end def get_response_BERT_answer_concat(self, q)
    
    def get_response_BERT_answer_para_concat(self, q):

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
        tf_idf_time = endTimeForTFIDF - timeForTFIDF
        print('Total time for TFIDF: {}'.format(tf_idf_time))
       
        question_answer_dict_list = list()
        response_answer_list = list()

        for paragraph in paras:
            question_answer_dict_list.append({'question': q, 'context': paragraph})

        scoreTime = time.time()
        for question in question_answer_dict_list:
            response = self.nlp(question)
            # print(response['answer'])
            # ************************************** Concatenated answers ************************************
            response_answer_list.append(response['answer'])
            # ************************************** Concatenated answers ************************************
            question['score'] = response['score']
        endScoreTime = time.time()
        total_score_time = endScoreTime - scoreTime
        print('Total time for scoring: {}'.format(total_score_time))
        
        # We want to get the list in descending order from best confidence score to worst.
        question_answer_dict_list_sorted = sorted(question_answer_dict_list, key = lambda i: i['score'], reverse=True)
        
        answer = "No answer yet"
        
        # *********************************************** Concatenated answers *******************************************************************************
        # The following lines are for testing purposes. They concatenate the context of the best spans of the five answers produced by  the BERT
        # confidence scores.
        concatenated_answers = ''
        for responseAnswer in response_answer_list:
            concatenated_answers = concatenated_answers + ' ' + responseAnswer

        print('\nConcatenated answers:\n{}\n'.format(concatenated_answers))
        
        # We will pass the concatenated_answers to BART along with the context associated with the highest score
        top_para = "question: " + q + " context: " + concatenated_answers + question_answer_dict_list_sorted[0]['context']
        
        # *********************************************** Concatenated answers end ***************************************************************************

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
        answer = qa_s2s_generate(top_para, self.bart_model, self.bart_tokenizer,num_answers=1,num_beams=8, min_len=96, max_len=256, max_input_length=1024, device='cuda:0')[0]
        # tf.get_variable_scope().reuse_variables()
        endAnswerTime = time.time()
        tf.get_variable_scope().reuse_variables()
        total_answer_time = endAnswerTime - answerTime
        print('\nTotal time to generate answer: {}\n'.format(total_answer_time))
        
        timeDict = {"tf_idf": tf_idf_time, "confidence_scores": total_score_time, "answer": total_answer_time}

        return answer, timeDict

    # end def get_response_BERT_answer_concat(self, q)
    
    def get_response_BERT_two_answer_context(self, q):

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
        top_para = "question: " + q + " context: " + question_answer_dict_list_sorted[0]['answer'] + ' ' + question_answer_dict_list_sorted[1]['answer'] + ' ' + question_answer_dict_list_sorted[0]['context']
        
        
        # print("\nAnswer by TriviQA:\n")
        # #print("Paragraph Order:" + str(best_paras))
        # print("Best Paragraph: " + str(best_para))
        # #print("Best span: " + str(best_spans[best_para]))
        # print("Answer text: " + " ".join(context[best_para][best_spans[best_para][0]:best_spans[best_para][1]+1]))
        # #print("Confidence: " + str(conf[best_para]))
        top_file = open("/home/bdlabucdenver/Top_para.txt",'w')
        top_file.write(top_para)
        top_file.close()
        
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
        
        timeDict = {"tf_idf": tf_idf_time, "confidence_scores": confidence_score_time, "answer": total_answer_time}
        
        return answer, timeDict

    # end def get_response_BERT_answer_concat(self, q)

    def get_response_10_concat_spans(self, q):

        start = time.time()
        timeForTFIDF = time.time()
        question = self.tokenizer.tokenize_paragraph_flat(q)  # List of words
        # Now select the top paragraphs using a `ParagraphFilter`
        if len(self.documents) == 1:
            # Use TF-IDF to select top paragraphs from the document
            selector = TopTfIdf(NltkPlusStopWords(True), n_to_select=10)
            context = selector.prune(question, self.documents[0])
        else:
            # Use a linear classifier to select top paragraphs among all the documents
            selector = ShallowOpenWebRanker(n_to_select=10)
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
        
        concatenated_spans = ''
        # concatenate the 10 answers from the best contexts
        for response in question_answer_dict_list_sorted:
            concatenated_spans = concatenated_spans + ' ' + response['answer']
        
        # We will pass the top two answers along with the highest scored context to BART
        top_para = "question: " + q + " context: " + concatenated_spans

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
        answer = qa_s2s_generate(top_para, self.bart_model, self.bart_tokenizer,num_answers=1,num_beams=8, min_len=96, max_len=256, max_input_length=1024, device='cuda:0')[0]
        # tf.get_variable_scope().reuse_variables()
        endAnswerTime = time.time()
        tf.get_variable_scope().reuse_variables()
        print('\nTotal time to generate answer: {}\n'.format(endAnswerTime - answerTime))

        return answer

    # end def get_response_BERT_answer_concat(self, q)
