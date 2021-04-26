Each folder is named for the model that was used in the experiment. distillbart, gpt2, and T5 are text 
generation models. iLFQA works by calculating the TF-IDF for each paragraph in the documents relevant to
a question. After five appropriate paragraphs are collected, they are each assigned confidence scores
using roberta. 

Each folder contains a number .xlsx files that contain the results of the experiments carried out by
the corresponding source code. Each results folder within this directory contains a readme that
has a link to the information about specific modesls and the corresponding methods that were used 
to perform the tests.

All results of the experiments folders are reporducable with the source code contained in each model 
folder.

Each model folder has three sub-folders:

	source: The source code that will allow you to reproduce results in results folder

	results: The results produced from running the code in source.

	bert_scores: The result of using BERTScore to evaluate the raw results of the source code.
	In order to score a raw results folder, follow the instructions contained in the BERTScore folder.
	
BERTScore:

	Contains the source code necessary to reproduce BERTScore results from each raw results file.
	
human_evaluations:

	Contains human evaluated responses for two verison of iTA. The original implemntation as well as
	the implementation that was finalized for the paper.
	
		get_respopnse_human_eval.xlsx: Human evaluated responses from original implementation of iTA.
		Evaluation performed by paper authors.
		
		
