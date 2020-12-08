import time
import collections
import numpy
from itertools import groupby
from collections import defaultdict     

class MLTEvaluator(object):
	def __init__(self, config):
		self.config = config
		self.cost_sum = 0.0
		self.sentence_count = 0.0
		self.sentence_correct_binary = 0.0
		self.sentence_predicted = 0.0
		self.sentence_correct = 0.0
		self.sentence_total = 0.0

		self.token_ap_sum = []
		self.token_predicted = []
		self.token_correct = []
		self.token_total = []

		self.entity_level_token_predicted = []
		self.entity_level_token_correct = []
		self.entity_level_token_total = []

		self.true_label_chunks = []
		self.predicted_label_chunks = []
		self.correct_label_chunks = []

		self.start_time = time.time()


	def get_chunk_type(self, tok, idx_to_tag):
		"""
		Args:
			tok: id of token, ex 4
			idx_to_tag: dictionary {4: "B-PER", ...}
		Returns:
			tuple: "B", "PER"
		"""
		tag_name = idx_to_tag[tok]
		tag_class = tag_name.split('-')[0]
		tag_type = tag_name.split('-')[-1]
		return tag_class, tag_type


	def get_chunks(self, seq, tags):
		"""Given a sequence of tags, group entities and their position
		Args:
			seq: [4, 4, 0, 0, ...] sequence of labels
			tags: dict["O"] = 4
		Returns:
			list of (chunk_type, chunk_start, chunk_end)
		Example:
			seq = [4, 5, 0, 3]
			tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
			result = [("PER", 0, 2), ("LOC", 3, 4)]
		"""
		default = 0
		idx_to_tag = {idx: tag for tag, idx in tags.items()}
		chunks = []
		chunk_type, chunk_start = None, None
		for i, tok in enumerate(seq):
			# End of a chunk 1
			if tok == default and chunk_type is not None:
				# Add a chunk.
				chunk = (chunk_type, chunk_start, i)
				chunks.append(chunk)
				chunk_type, chunk_start = None, None
			# End of a chunk + start of a chunk!
			elif tok != default:
				tok_chunk_class, tok_chunk_type = self.get_chunk_type(tok, idx_to_tag)
				if chunk_type is None:
					chunk_type, chunk_start = tok_chunk_type, i
				elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
					chunk = (chunk_type, chunk_start, i)
					chunks.append(chunk)
					chunk_type, chunk_start = tok_chunk_type, i
			else:
				pass
		# end condition
		if chunk_type is not None:
			chunk = (chunk_type, chunk_start, len(seq))
			chunks.append(chunk)
		return chunks


	def calculate_ap(self, true_labels, predicted_scores):
		assert(len(true_labels) == len(predicted_scores))
		indices = numpy.argsort(numpy.array(predicted_scores))[::-1]

		summed, correct, total = 0.0, 0.0, 0.0
		for index in indices:
			total += 1.0
			if true_labels[index] > 0:      
				correct += 1.0
				summed += correct / total
		return (summed / correct) if correct > 0.0 else 0.0


	def append_token_data_for_sentence(self, index, true_labels, token_scores, dict_tags):
		if len(self.token_ap_sum) <= index:
			self.token_ap_sum.append(0.0)
			self.token_predicted.append(0.0)
			self.token_correct.append(0.0)
			self.token_total.append(0.0)

		ap = self.calculate_ap(true_labels, token_scores[:len(true_labels)])
		self.token_ap_sum[index] += ap  

		# #for relaxed word-level evaluation
		for i in range(len(true_labels)):
			if true_labels[i] > 0:      
				self.token_total[index] += 1.0
				self.entity_level_token_total.append([index, true_labels[i], 1])
			if token_scores[i] > 0:
				self.token_predicted[index] += 1.0
				self.entity_level_token_predicted.append([index, token_scores[i], 1])
			if true_labels[i] > 0 and token_scores[i] > 0 and true_labels[i] == token_scores[i]:
				self.token_correct[index] += 1.0
				self.entity_level_token_correct.append([index, true_labels[i], token_scores[i], 1])

		# #for strict phrase-level evaluation
		sentence_true_label_chunks = []
		sentence_predicted_label_chunks = []
		sentence_correct_label_chunks = []
		sentence_true_label_chunks = set(self.get_chunks(true_labels, dict_tags))
		sentence_predicted_label_chunks = set(self.get_chunks(token_scores, dict_tags))
		sentence_correct_label_chunks = set(sentence_true_label_chunks & sentence_predicted_label_chunks)

		self.true_label_chunks += sentence_true_label_chunks
		self.predicted_label_chunks += sentence_predicted_label_chunks
		self.correct_label_chunks += sentence_correct_label_chunks	


	def append_data(self, cost, batch, sentence_scores, token_scores_list, token_probs):
		assert(len(self.token_ap_sum) == 0 or len(self.token_ap_sum) == len(token_scores_list))
		self.cost_sum += cost
		for i in range(len(batch)):
			self.sentence_count += 1.0

			# Make a dictionary of tags. Make sure there is a file "tags.txt" with all the tags in the dataset. For binary it will be "O" and "tagname". For mulit-class it will me many. Also, make sure the tags are in order.
			dict_tags = dict()
			with open(self.config["filename_tags"]) as f:
				for idx, word in enumerate(f):
					word = word.strip()
					dict_tags[word] = idx+1     # we have (idx+1) because we deleted defalut label "O" from the tags file

			true_labels = []
			for j in range(len(batch[i])):             
				if batch[i][j][-1] == self.config["default_label"]:
					true_labels.append(0)
				else:            
					for tag, idx in dict_tags.items():
						if batch[i][j][-1] == tag:
							true_labels.append(idx)                       

			count_interesting_labels = numpy.array([1.0 if batch[i][j][-1] in ['B-ADE', 'I-ADE', 'ADE'] else 0.0 for j in range(len(batch[i]))]).sum()
			if (count_interesting_labels == 0.0 and sentence_scores[i] < 0.5) or (count_interesting_labels > 0.0 and sentence_scores[i] >= 0.5):
				self.sentence_correct_binary += 1.0
			if sentence_scores[i] >= 0.5:
				self.sentence_predicted += 1.0
			if count_interesting_labels > 0.0:
				self.sentence_total += 1.0
			if count_interesting_labels > 0.0 and sentence_scores[i] >= 0.5:
				self.sentence_correct += 1.0
									
			for k in range(len(token_scores_list)):
				self.append_token_data_for_sentence(k, true_labels, token_scores_list[k][i], dict_tags)


	def get_results(self, name):
		print ("\n")
		p = (float(self.sentence_correct) / float(self.sentence_predicted)) if (self.sentence_predicted > 0.0) else 0.0
		r = (float(self.sentence_correct) / float(self.sentence_total)) if (self.sentence_total > 0.0) else 0.0
		f = (2.0 * p * r / (p + r)) if (p+r > 0.0) else 0.0
		f05 = ((1.0 + 0.5*0.5) * p * r / ((0.5*0.5 * p) + r)) if (((0.5*0.5 * p) + r) > 0.0) else 0.0

		results = collections.OrderedDict()
		results[name + "_cost_sum"] = self.cost_sum
		results[name + "_cost_avg"] = self.cost_sum / float(self.sentence_count)
		results[name + "_sent_count"] = self.sentence_count
		results[name + "_sent_predicted"] = self.sentence_predicted
		results[name + "_sent_correct"] = self.sentence_correct
		results[name + "_sent_total"] = self.sentence_total
		results[name + "_sent_p"] = p
		results[name + "_sent_r"] = r
		results[name + "_sent_f"] = f
		results[name + "_sent_f05"] = f05
		results[name + "_sent_correct_binary"] = self.sentence_correct_binary
		results[name + "_sent_accuracy_binary"] = self.sentence_correct_binary / float(self.sentence_count)

		# ####### strict_evaluation ##########
		p_all = (len(self.correct_label_chunks)) / float(len(self.predicted_label_chunks)) if (len(self.predicted_label_chunks) > 0.0) else 0.0
		r_all = (len(self.correct_label_chunks)) / float(len(self.true_label_chunks)) if (len(self.true_label_chunks) > 0.0) else 0.0
		f_all = (2.0 * p_all * r_all / (p_all + r_all)) if (p_all+r_all > 0.0) else 0.0
		f05_all = ((1.0 + 0.5*0.5) * p_all * r_all / ((0.5*0.5 * p_all) + r_all)) if (((0.5*0.5 * p_all) + r_all) > 0.0) else 0.0  

		results[name + "_tok_predicted"] = len(self.predicted_label_chunks)
		results[name + "_tok_correct"] = len(self.correct_label_chunks)
		results[name + "_tok_total"] = len(self.true_label_chunks)
		
		results[name + "_tok_"+"_p"] = p_all
		results[name + "_tok_"+"_r"] = r_all
		results[name + "_tok_"+"_f"] = f_all
		results[name + "_tok_"+"_f05"] = f05_all

		grouped_correct_preds = defaultdict(list)  
		for i in self.correct_label_chunks:
			grouped_correct_preds[i[0]].append(i)
		for key in grouped_correct_preds:    # key is the entity name 
			k = 0
			for j in grouped_correct_preds[key]:
				k += 1 

		grouped_total_preds = defaultdict(list)  
		for i in self.predicted_label_chunks:
			grouped_total_preds[i[0]].append(i)
		for key in grouped_total_preds:    # key is the entity name 
			k = 0
			for j in grouped_total_preds[key]:
				k += 1 

		grouped_total_truelabels = defaultdict(list)  
		for i in self.true_label_chunks:
			grouped_total_truelabels[i[0]].append(i)
		for key in grouped_total_truelabels:    # key is the entity name 
			k = 0
			for j in grouped_total_truelabels[key]:
				k += 1         

		for key in grouped_total_truelabels:    # key is the entity name 
			truelabels = total_preds = correct_preds = 0
			for j in grouped_total_truelabels[key]:
				truelabels += 1       					  
			for j in grouped_total_preds[key]:
				total_preds += 1  
			for j in grouped_correct_preds[key]:
				correct_preds += 1   
			p_group = float(correct_preds) / float(total_preds) if float(total_preds) > 0.0 else 0.0
			r_group = float(correct_preds) / float(truelabels) if float(truelabels) > 0.0 else 0.0
			f_group = (2.0 * p_group * r_group / (p_group + r_group)) if (p_group+r_group > 0.0) else 0.0
			f05_group = ((1.0 + 0.5*0.5) * p_group * r_group / ((0.5*0.5 * p_group) + r_group)) if (((0.5*0.5 * p_group) + r_group) > 0.0) else 0.0  
			# print (str(key), ":", p, r, f)

			results[name + "_tok_"+str(key)+"_predicted"] = total_preds
			results[name + "_tok_"+str(key)+"_correct"] = correct_preds
			results[name + "_tok_"+str(key)+"_total"] = truelabels

			results[name + "_tok_"+str(key)+"_p"] = p_group
			results[name + "_tok_"+str(key)+"_r"] = r_group
			results[name + "_tok_"+str(key)+"_f"] = f_group
			results[name + "_tok_"+str(key)+"_f05"] = f05_group

		results[name + "_time"] = float(time.time()) - float(self.start_time)

		return results
