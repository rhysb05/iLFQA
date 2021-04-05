Each folder is named for the model that was used in the experiment. distillbart, gpt2, and T5 are text 
generation models. iTA works by calculating the TF-IDF for each paragraph in the documents relevant to
a question. After five appropriate paragraphs are collected, they are each assigned confidence scores
using roberta. 

Each folder contains a number .xlsx files that contain the results of the experiments carried out by
the corresponding source code. Each results folder within this directory contains a readme that
has a link to the information about specific modesls and the corresponding methods that were used 
to perform the tests.

All results of the experiments folders are reporducable with the source code contained in each model 
folder.

	roberta: Contains results of experiments that altered the length and semantic meaning of 
	the context in order to evaluate performance. These methods were chosen heurisitcally during the
	early days of using roberta to assign confidence scores. 
	
	distillbart, gpt2, T5: Each a text generation model that was tested as a possible candidate for iTA.
	