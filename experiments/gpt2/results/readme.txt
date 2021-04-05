//All questions from the file QA_rephrase.xlsx


QA_Results_gpt2_2Answer_Concat:
	
	Description:
	Contains results of multi_question test where best two spans determined by Roberta
	are appended to the highest scored context.
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	https://huggingface.co/gpt2
	
QA_Results_gpt2_2Answer_Concat_no_seed:
	
	Description:
	Contains results of multi_question test where the four best contexts are concatenated together and
	submitted to the text generation model.
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	https://huggingface.co/gpt2
	
