//All questions from the file QA_rephrase.xlsx


QA_Results_2Answer_Concat_rephrase:
	
	Description:
	Contains results of multi_question test where best two spans determined by Roberta
	are appended to the highest scored context.
	
	Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	https://huggingface.co/yjernite/bart_eli5
	
QA_Results_BERT:
	
	Description:
	Contains results of multi_question test where two best scored contexts are concatenated and
	sent to the BART model.
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	https://huggingface.co/yjernite/bart_eli5
	
QA_Results_BERT_5_best_spans_concat_spans:
	
	Description:
	Contains results of multi_question test where the 5 best spans are concatenated and sent to
	BART model.
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	https://huggingface.co/yjernite/bart_eli5
	
QA_Results_BERT_5_best_spans_top_para:
	
	Description:
	Contains results of multi_question test where the 5 best spans and best context are concatenated
	and sent to BART model.
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	https://huggingface.co/yjernite/bart_eli5
	
QA_Results_BERT_10_answer_concat:
	
	Description:
	Contains results of multi_question test where the 10 best spans are concatenated and 
	those answers are submitted to BART model.
	
	Confidence scoring Model:
	"deepset/roberta-base-squad2"
	https://huggingface.co/deepset/roberta-base-squad2
	
	Text2Text model:
	https://huggingface.co/yjernite/bart_eli5