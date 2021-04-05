//All questions from the file QA_rephrase.xlsx


five_answer_best_context_concat.xlsx:
	
	Description:
	Contains results of multi_question test where the five highest scored spans (determined by roberta)
	are concatenated with the highest scored context and sent to the BART  model. Obtain results
	by running multi_question_pegasus_summ.py with line 45 as follows:
	
	answer, time_dict, context = res.two_answer_best_context_concat(d[0])
	
	Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	https://huggingface.co/yjernite/bart_eli5
	
four_best_context_concat.xlsx:
	
	Description:
	Contains results of multi_question test where only the four highest scoring contexts
	(determined by roberta) are concatenated together and sent to the BART model for text
	generation. Obtain results by running multi_question_pegasus_summ.py with line 45 as follows:
	
	answer, time_dict, context = res.four_best_context_concat(d[0])
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	https://huggingface.co/yjernite/bart_eli5
	
two_answer_best_context_concat.xlsx:
	
	Description:
	Contains results of multi_question test where the second and third best answer spans
	are concatenated with the best scored context (paragraph). We choose the second and third best spans
	because the spans are exceprts of text contained in the context. We do not want to simply repeat
	certain n-grams. Obtain results by running multi_question_pegasus_summ.py with line 45 as follows:
	
	answer, time_dict, context = res.two_answer_best_context_concat(d[0])
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	https://huggingface.co/yjernite/bart_eli5
	
two_answer_two_context_concat.xlsx:
	
	Description:
	Contains results of multi_question test where the third and fourth best answer spans
	are concatenated with the two higest scored contexts (paragraph). We choose the second and third best spans
	because the spans are exceprts of text contained in the context. We do not want to simply repeat
	certain n-grams. Obtain results by running multi_question_pegasus_summ.py with line 45 as follows:
	
	answer, time_dict, context = res.two_answer_two_context_concat(d[0])
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	https://huggingface.co/yjernite/bart_eli5
	