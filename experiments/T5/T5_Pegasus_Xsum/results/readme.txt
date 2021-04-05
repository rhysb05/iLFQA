//All questions from the file QA_rephrase.xlsx


four_best_context_concat.xlsx:
	
	Description:
	Contains results of multi_question test where the four highest confidence
	paragraphs are concatenated and sent to T5 pegasus-xsum. Obtain results
	by running multi_question_pegasus_summ.py with line 45 as follows:
	
	answer, time_dict, context = res.four_best_context_concat(d[0])
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	'google/pegasus-xsum'
	https://huggingface.co/google/pegasus-xsum
	
two_answer_two_context_concat.xlsx:
	
	Description:
	Contains results of multi_question test where the two best spans determined by Roberta
	are appended to the two best scored contexts and sent to T5 pegasus-xsum. Obtain results
	by running multi_question_pegasus_summ.py with line 45 as follows:
	
	answer, time_dict, context = res.two_answer_two_context_concat(d[0])
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	'google/pegasus-xsum'
	https://huggingface.co/google/pegasus-xsum
	
two_answer_best_context_concat.xlsx:
	
	Description:
	Contains results of multi_question test where the second and third highest scored
	spans and the highest scored paragraph are concatenated together and sent to T5.
	Obtain results by running multi_question_pegasus_summ.py with line 45 as follows:
	
	answer, time_dict, context = res.two_answer_best_context_concat(d[0])
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	'google/pegasus-xsum'
	https://huggingface.co/google/pegasus-xsum
	
