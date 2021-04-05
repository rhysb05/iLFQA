//All questions from the file QA_rephrase.xlsx


four_best_context_concat.xlsx:
	
	Description:
	Contains results of multi_question test where the four best spans determined by Roberta
	are appended to the highest scored context and sent to T5 pegasus-xsum. Obtain results
	by running multi_question_pegasus_summ.py with line 45 as follows:
	
	answer, time_dict = res.four_best_context_concat(d[0])
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	'google/pegasus-xsum'
	https://huggingface.co/google/pegasus-xsum
	
two_answer_two_context_concat.xlsx:
	
	Description:
	Contains results of multi_question test where the four best contexts are concatenated together and
	submitted to the text generation model.
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	'google/pegasus-xsum'
	https://huggingface.co/google/pegasus-xsum
	
five_answer_best_context_concat.xlsx:
	
	Description:
	Contains results of multi_question test where the four best contexts are concatenated together and
	submitted to the text generation model.
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	'google/pegasus-xsum'
	https://huggingface.co/google/pegasus-xsum
	
