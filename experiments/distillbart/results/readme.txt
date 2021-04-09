//All questions from the file QA_rephrase.xlsx


five_answer_best_context_concat:
	
	Description:
	Contains results of multi_question test where the five highest scored spans (determined by roberta)
	are concatenated with the highest scored context and sent to the distillbart model. Obtain results
	by running multi_question_BERT_distilbart.py with line 46 as follows:
	
	answer, time_dict, context = res.five_answer_best_context_concat(d[0])
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	"sshleifer/distilbart-cnn-6-6"
	https://huggingface.co/sshleifer/distilbart-cnn-6-6
	
four_best_context_concat:
	
	Description:
	Contains results of multi_question test where the four highest scored paragraphs
	are scored and sent to the distillbart model. Obtain results
	by running running multi_question_BERT_distilbart.py with line 46 as follows:
	
	answer, time_dict, context = res.four_best_context_concat(d[0])
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	"sshleifer/distilbart-cnn-6-6"
	https://huggingface.co/sshleifer/distilbart-cnn-6-6
	
two_answer_best_context_concat:
	
	Description:
	Contains results of multi_question test where the two highest scored spans
	and the higest scored paragraph are concatenated and sent to the distillbart model. 
	Obtain results by running running multi_question_BERT_distilbart.py with line 46 as follows:
	
	answer, time_dict, context = res.two_answer_best_context_concat(d[0])
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	"sshleifer/distilbart-cnn-6-6"
	https://huggingface.co/sshleifer/distilbart-cnn-6-6
	
two_answer_two_context_concat:
	
	Description:
	Contains results of multi_question test where the two highest scored spans
	and the two higest scored paragraph are concatenated and sent to the gpt2  model. 
	Obtain results by running running multi_question_BERT_distilbart.py with line 46 as follows:
	
	answer, time_dict, context = res.two_answer_best_context_concat(d[0])
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	"sshleifer/distilbart-cnn-6-6"
	https://huggingface.co/sshleifer/distilbart-cnn-6-6
	
