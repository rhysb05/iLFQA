//All questions from the file QA_rephrase.xlsx


QA_Results_T5_pegasus_2_concat_para:
	
	Description:
	Contains results of multi_question test where best two spans determined by Roberta
	are appended to the highest scored context.
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	'google/pegasus-xsum'
	https://huggingface.co/google/pegasus-xsum
	
QA_Results_T5_pegasus_4_para_concat_summ:
	
	Description:
	Contains results of multi_question test where the four best contexts are concatenated together and
	submitted to the text generation model.
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	'google/pegasus-xsum'
	https://huggingface.co/google/pegasus-xsum
	
