import sys
import functools

try:
	import ConfigParser as configparser
except:
	import configparser

from model import MLTModel
from evaluator import MLTEvaluator
from experiment import read_input_files


if __name__ == "__main__":
	model = MLTModel.load(sys.argv[1])
	data = read_input_files(sys.argv[2], -1)
	batch_size = 32
	evaluator = MLTEvaluator(model.config)


	# # Make a dictionary of tags. Make sure there is a file "tags.txt" with all the tags in the dataset. For binary it will be "O" and "tagname". For mulit-class it will me many. Also, make sure the tags are in order.
	dict_tags = dict()
	filename_tags = "tags.txt" #add correct path to the tags file

	with open(filename_tags) as f:
		for idx, word in enumerate(f):
			word = word.strip()
			dict_tags[word] = idx+1     # we have (idx+1) because we deleted defalut label "O" from the tags file
	print("sw_print:dict_tags", dict_tags.items())

	print_ALL_weights = True
	print_no_attention_sentLast = False

	if print_ALL_weights == True:
		print("Word" + "\t" + "true_label(text)" + "true_label(num)" + "\t" + "true_label_predicted_proba" + "\t" + "model_predicted_label(num)" + "\t" + "model_predicted_label_proba" + "\t" + "sentence_predicted_proba" + "\t" + "unsup_weights" + "\t" + "selective_weights" + "\t" + "indicative_weights")		
	if print_no_attention_sentLast == True:		
		print("Word" + "\t" + "true_label(text)" + "true_label(num)" + "\t" + "sum of model_predicted_proba(B-ADE + I-ADE)" + "\t" + "sentence_predicted_proba")	

	for i in range(0, len(data), batch_size):
		batch = data[i:i+batch_size]
		cost, sentence_scores, token_scores_list, token_probs, token_probs_all_labels, unsup_weights, selective_weights, indicative_weights = model.process_batch(batch, False, 0.0)		

		for j in range(len(batch)):
			for k in range(len(batch[j])):

				if batch[j][k][-1] == 'O':
					label_index = 0
				else:            
					for tag, idx in dict_tags.items():
						if batch[j][k][-1] == tag:
							label_index = idx  

				if print_ALL_weights == True:
					print(" ".join([str(x) for x in batch[j][k]]) + "\t" + str(label_index) + "\t" + str(token_probs_all_labels[j][k][label_index]) + "\t" + str(token_scores_list[0][j][k]) + "\t" + str(token_probs[0][j][k]) + "\t" + str(sentence_scores[j]) + "\t" + str(unsup_weights[j][k]) + "\t" + str(selective_weights[j][k]) + "\t" + str(indicative_weights[j][k])) 				
				if print_no_attention_sentLast == True:
					print(" ".join([str(x) for x in batch[j][k]]) + "\t" + str(label_index)  + "\t" + str(sentence_scores[j]))         

			print("")   
		

		evaluator.append_data(cost, batch, sentence_scores, token_scores_list, token_probs)

	results = evaluator.get_results("test")
	for key in results:
		sys.stderr.write(key + ": " + str(results[key]) + "\n")
