import itertools, collections
import tensorflow as tf
import re
import ast
import numpy
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

try:
	import cPickle as pickle
except:
	import pickle

class MLTModel(object):
	def __init__(self, config):
		self.config = config

		self.UNK = "<unk>"
		self.CUNK = "<cunk>"

		self.word2id = None
		self.char2id = None
		self.singletons = None
		self.dict_tags = None   
		self.print_order = 0	


	def build_vocabs(self, data_train, data_dev, data_test, embedding_path=None):
		data_source = list(data_train)
		if self.config["vocab_include_devtest"]:
			if data_dev != None:
				data_source += data_dev
			if data_test != None:
				data_source += data_test

		char_counter = collections.Counter()
		for sentence in data_source:
			for word in sentence:
				char_counter.update(word[0])
		self.char2id = collections.OrderedDict([(self.CUNK, 0)])
		for char, count in char_counter.most_common():
			if char not in self.char2id:
				self.char2id[char] = len(self.char2id)

		word_counter = collections.Counter()
		for sentence in data_source:
			for word in sentence:
				w = word[0]
				if self.config["lowercase"] == True:
					w = w.lower()
				if self.config["replace_digits"] == True:
					w = re.sub(r'\d', '0', w)
				word_counter[w] += 1

		self.word2id = collections.OrderedDict([(self.UNK, 0)])
		for word, count in word_counter.most_common():
			if self.config["min_word_freq"] <= 0 or count >= self.config["min_word_freq"]:
				if word not in self.word2id:
					self.word2id[word] = len(self.word2id)

		self.singletons = set([word for word in word_counter if word_counter[word] == 1])

		if embedding_path != None and self.config["vocab_only_embedded"] == True:
			self.embedding_vocab = set([self.UNK])
			with open(embedding_path, 'r') as f:
				for line in f:
					line_parts = line.strip().split()
					if len(line_parts) <= 2:
						continue
					w = line_parts[0]
					if self.config["lowercase"] == True:
						w = w.lower()
					if self.config["replace_digits"] == True:
						w = re.sub(r'\d', '0', w)
					self.embedding_vocab.add(w)
			word2id_revised = collections.OrderedDict()
			for word in self.word2id:
				if word in embedding_vocab and word not in word2id_revised:
					word2id_revised[word] = len(word2id_revised)
			self.word2id = word2id_revised


		# Make a dictionary of tags. Make sure there is a file "tags.txt" with all the tags in the dataset. For binary it will be "O" and "tagname". For mulit-class it will me many. Also, make sure the tags are in order.
		self.dict_tags = dict()
		with open(self.config["filename_tags"]) as f:
			for idx, word in enumerate(f):
				word = word.strip()
				self.dict_tags[word] = idx+1      # we have (idx+1) because we deleted defalut label "O" from the tags file

		print("n_words: " + str(len(self.word2id)))
		print("n_chars: " + str(len(self.char2id)))
		print("n_singletons: " + str(len(self.singletons)))


	def da_unsupervised_self_attention(self,lstm_outputs):			
		unsup_attention_evidence = tf.layers.dense(lstm_outputs, self.config["attention_evidence_size"], activation=tf.tanh, kernel_initializer=self.initializer)     
		unsup_attention_weights = tf.layers.dense(unsup_attention_evidence, 1, activation=None, kernel_initializer=self.initializer)             
		unsup_attention_weights = tf.reshape(unsup_attention_weights, shape=tf.shape(self.word_ids))
		unsup_attention_weights = tf.sigmoid(unsup_attention_weights)
		unsup_attention_weights = tf.where(tf.sequence_mask(self.sentence_lengths), unsup_attention_weights, tf.zeros_like(unsup_attention_weights))
		unsup_attention_weights = unsup_attention_weights / tf.reduce_sum(unsup_attention_weights, 1, keep_dims=True)   
		processed_tensor = tf.reduce_sum(lstm_outputs * unsup_attention_weights[:,:,numpy.newaxis], 1)    
		return processed_tensor, unsup_attention_weights


	def da_predicted_label_weight(self,lstm_outputs,processed_tensor):
		if self.config["mask0_predicted_label_weight"] == True:
			unmasked_attention_weights = self.nonZeroClass_prediction_probs		
		else:
			unmasked_attention_weights = self.prediction_probs		
		unmasked_attention_weights = tf.where(tf.sequence_mask(self.sentence_lengths), unmasked_attention_weights, tf.zeros_like(unmasked_attention_weights))
		unmasked_attention_weights = unmasked_attention_weights / tf.reduce_sum(unmasked_attention_weights, 1, keep_dims=True)   
		beta_attention_weights = tf.reduce_sum(lstm_outputs * unmasked_attention_weights[:,:,numpy.newaxis], 1)    

		if processed_tensor is None:
			processed_tensor = beta_attention_weights
		else:
			if self.config["double_attention_predicted_label_weight_jointype"] == "add":
				processed_tensor = tf.add(processed_tensor, beta_attention_weights)
			elif self.config["double_attention_predicted_label_weight_jointype"] == "concat":
				processed_tensor = tf.concat([processed_tensor,beta_attention_weights],-1)
			elif self.config["double_attention_predicted_label_weight_jointype"] == "separate":	
				alpha_processed_tensor = tf.layers.dense(processed_tensor, self.config["hidden_layer_size"], activation=tf.tanh, kernel_initializer=self.initializer)
				beta_processed_tensor = tf.layers.dense(beta_attention_weights, self.config["hidden_layer_size"], activation=tf.tanh, kernel_initializer=self.initializer)
				processed_tensor = tf.concat([alpha_processed_tensor,beta_processed_tensor],-1)
		return processed_tensor, unmasked_attention_weights


	def da_predicted_label(self,attention_weights_argmax,processed_tensor):
		oh_predicted_label = tf.one_hot(attention_weights_argmax, self.config["ntags"])
		attention_from_label = tf.cast((tf.cast(tf.reduce_sum(oh_predicted_label,1), tf.bool)), tf.float32)

		if self.config["mask_attention_from_label"] == True: 
			attention_from_label_col_to_zero = [0]	# column numbers you want to be zeroed out
			attention_from_label_tnsr_shape = tf.shape(attention_from_label)
			attention_from_label_mask = [tf.one_hot(col_num*tf.ones((attention_from_label_tnsr_shape[0], ), dtype=tf.int32), attention_from_label_tnsr_shape[-1]) for col_num in attention_from_label_col_to_zero]
			attention_from_label_mask = tf.reduce_sum(attention_from_label_mask, axis=0)
			attention_from_label_mask = tf.cast(tf.logical_not(tf.cast(attention_from_label_mask, tf.bool)), tf.float32)
			attention_from_label = (attention_from_label * attention_from_label_mask)
		processed_tensor = tf.concat([processed_tensor,attention_from_label],-1) 	
		return processed_tensor
		

	def construct_network(self):
		self.word_ids = tf.placeholder(tf.int32, [None, None], name="word_ids")
		self.char_ids = tf.placeholder(tf.int32, [None, None, None], name="char_ids")
		self.sentence_lengths = tf.placeholder(tf.int32, [None], name="sentence_lengths")
		self.word_lengths = tf.placeholder(tf.int32, [None, None], name="word_lengths")
		self.sentence_labels = tf.placeholder(tf.float32, [None,], name="sentence_labels")
		self.word_labels = tf.placeholder(tf.int32, [None,None], name="word_labels")         
		self.word_labels_oh = tf.placeholder(tf.float32, [None,None,None], name="word_labels_oh") 
		self.word_objective_weights = tf.placeholder(tf.float32, [None,None], name="word_objective_weights")
		self.sentence_objective_weights = tf.placeholder(tf.float32, [None], name="sentence_objective_weights")
		self.learningrate = tf.placeholder(tf.float32, name="learningrate")
		self.is_training = tf.placeholder(tf.int32, name="is_training")
		self.ones_tensor = tf.placeholder(tf.float32, [None,None,None], name="ones_tensor")		
		self.unsup_attention_weights_normalised = tf.zeros_like(self.word_ids, dtype=tf.float32)	
		self.selective_attention_weights_normalized = tf.zeros_like(self.word_ids, dtype=tf.float32)	
		self.indicative_attention_weights_normalized = tf.zeros_like(self.word_ids, dtype=tf.float32)	

		self.loss = 0.0
		input_tensor = None
		input_vector_size = 0

		self.initializer = None
		if self.config["initializer"] == "normal":
			self.initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1)
		elif self.config["initializer"] == "glorot":
			self.initializer = tf.glorot_uniform_initializer()
		elif self.config["initializer"] == "xavier":
			self.initializer = tf.glorot_normal_initializer()

		zeros_initializer = tf.zeros_initializer()

		self.word_embeddings = tf.get_variable("word_embeddings", 
			shape=[len(self.word2id), self.config["word_embedding_size"]], 
			initializer=(zeros_initializer if self.config["emb_initial_zero"] == True else self.initializer), 
			trainable=(True if self.config["train_embeddings"] == True else False))
		input_tensor = tf.nn.embedding_lookup(self.word_embeddings, self.word_ids)
		input_vector_size = self.config["word_embedding_size"]

		if self.config["char_embedding_size"] > 0 and self.config["char_recurrent_size"] > 0:
			with tf.variable_scope("chars"), tf.control_dependencies([tf.assert_equal(tf.shape(self.char_ids)[2], tf.reduce_max(self.word_lengths), message="Char dimensions don't match")]):
				self.char_embeddings = tf.get_variable("char_embeddings", 
					shape=[len(self.char2id), self.config["char_embedding_size"]], 
					initializer=self.initializer, 
					trainable=True)
				char_input_tensor = tf.nn.embedding_lookup(self.char_embeddings, self.char_ids)

				s = tf.shape(char_input_tensor)
				char_input_tensor = tf.reshape(char_input_tensor, shape=[s[0]*s[1], s[2], self.config["char_embedding_size"]])
				_word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

				char_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config["char_recurrent_size"], 
					use_peepholes=self.config["lstm_use_peepholes"], 
					state_is_tuple=True, 
					initializer=self.initializer,
					reuse=False)
				char_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config["char_recurrent_size"], 
					use_peepholes=self.config["lstm_use_peepholes"], 
					state_is_tuple=True, 
					initializer=self.initializer,
					reuse=False)

				char_lstm_outputs = tf.nn.bidirectional_dynamic_rnn(char_lstm_cell_fw, char_lstm_cell_bw, char_input_tensor, sequence_length=_word_lengths, dtype=tf.float32, time_major=False)
				_, ((_, char_output_fw), (_, char_output_bw)) = char_lstm_outputs
				char_output_tensor = tf.concat([char_output_fw, char_output_bw], axis=-1)
				char_output_tensor = tf.reshape(char_output_tensor, shape=[s[0], s[1], 2 * self.config["char_recurrent_size"]])
				char_output_vector_size = 2 * self.config["char_recurrent_size"]

				if self.config["lmcost_char_gamma"] > 0.0:
					self.loss += self.config["lmcost_char_gamma"] * self.construct_lmcost(char_output_tensor, char_output_tensor, self.sentence_lengths, self.word_ids, "separate", "lmcost_char_separate")
				if self.config["lmcost_joint_char_gamma"] > 0.0:
					self.loss += self.config["lmcost_joint_char_gamma"] * self.construct_lmcost(char_output_tensor, char_output_tensor, self.sentence_lengths, self.word_ids, "joint", "lmcost_char_joint")

				if self.config["char_hidden_layer_size"] > 0:
					char_output_tensor = tf.layers.dense(char_output_tensor, self.config["char_hidden_layer_size"], activation=tf.tanh, kernel_initializer=self.initializer)
					char_output_vector_size = self.config["char_hidden_layer_size"]

				if self.config["char_integration_method"] == "concat":
					input_tensor = tf.concat([input_tensor, char_output_tensor], axis=-1)
					input_vector_size += char_output_vector_size
				elif self.config["char_integration_method"] == "none":
					input_tensor = input_tensor
				else:
					raise ValueError("Unknown char integration method")

		self.word_representations = input_tensor

		dropout_input = self.config["dropout_input"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
		input_tensor =  tf.nn.dropout(input_tensor, dropout_input, name="dropout_word")

		word_lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], 
			use_peepholes=self.config["lstm_use_peepholes"], 
			state_is_tuple=True, 
			initializer=self.initializer,
			reuse=False)
		word_lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config["word_recurrent_size"], 
			use_peepholes=self.config["lstm_use_peepholes"], 
			state_is_tuple=True, 
			initializer=self.initializer,
			reuse=False)

		with tf.control_dependencies([tf.assert_equal(tf.shape(self.word_ids)[1], tf.reduce_max(self.sentence_lengths), message="Sentence dimensions don't match")]):
			(lstm_outputs_fw, lstm_outputs_bw), ((_, lstm_output_fw), (_, lstm_output_bw)) = tf.nn.bidirectional_dynamic_rnn(word_lstm_cell_fw, word_lstm_cell_bw, input_tensor, sequence_length=self.sentence_lengths, dtype=tf.float32, time_major=False)

		dropout_word_lstm = self.config["dropout_word_lstm"] * tf.cast(self.is_training, tf.float32) + (1.0 - tf.cast(self.is_training, tf.float32))
		lstm_outputs_fw =  tf.nn.dropout(lstm_outputs_fw, dropout_word_lstm, noise_shape=tf.convert_to_tensor([tf.shape(self.word_ids)[0],1,self.config["word_recurrent_size"]], dtype=tf.int32))
		lstm_outputs_bw =  tf.nn.dropout(lstm_outputs_bw, dropout_word_lstm, noise_shape=tf.convert_to_tensor([tf.shape(self.word_ids)[0],1,self.config["word_recurrent_size"]], dtype=tf.int32))
		lstm_outputs = tf.concat([lstm_outputs_fw, lstm_outputs_bw], -1)

		if self.config["whidden_layer_size"] > 0:
			lstm_outputs = tf.layers.dense(lstm_outputs, self.config["whidden_layer_size"], activation=tf.tanh, kernel_initializer=self.initializer)    

		self.lstm_outputs = lstm_outputs             

		lstm_output = tf.concat([lstm_output_fw, lstm_output_bw], -1)
		lstm_output = tf.nn.dropout(lstm_output, dropout_word_lstm)

		if self.config["sentence_composition"] == "last":
			processed_tensor = lstm_output       
			self.prediction_labels = tf.zeros_like(self.word_ids, dtype=tf.int32) 	
			self.token_probs_all_labels = tf.zeros_like(self.word_ids, dtype=tf.float32) 
			self.prediction_probs = tf.zeros_like(self.word_ids, dtype=tf.float32) 

		elif self.config["sentence_composition"] == "unsupervised_self_attention":
			with tf.variable_scope("unsupervised_self_attention"):
				processed_tensor, self.unsup_attention_weights_normalised = self.da_unsupervised_self_attention(lstm_outputs)
				self.prediction_labels = tf.zeros_like(self.word_ids, dtype=tf.int32) 	
				self.token_probs_all_labels = tf.zeros_like(self.word_ids, dtype=tf.float32) 
				self.prediction_probs = tf.zeros_like(self.word_ids, dtype=tf.float32) 

		elif self.config["sentence_composition"] == "attention":
			with tf.variable_scope("attention"):

				if self.config["double_attention_unsupervised_self_attention"] == True:
					unsup_processed_tensor, self.unsup_attention_weights_normalised = self.da_unsupervised_self_attention(lstm_outputs)
			
				attention_evidence = tf.layers.dense(lstm_outputs, self.config["attention_evidence_size"], activation=tf.tanh, kernel_initializer=self.initializer)     
				attention_weights = tf.layers.dense(attention_evidence, self.config["ntags"], activation=None, kernel_initializer=self.initializer)   

				if self.config["word_loss_function"] == "crossentropy":           
					word_objective_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=attention_weights, labels=self.word_labels)     

				if self.config["word_loss_function"] == "crossentropy":
					if self.config["attention_activation"] == "sharp":
						attention_weights = tf.exp(attention_weights)   
					elif self.config["attention_activation"] == "soft":     
						attention_weights = tf.sigmoid(attention_weights)   
					elif self.config["attention_activation"] == "multi":    
						attention_weights = tf.nn.softmax(attention_weights)                      
					elif self.config["attention_activation"] == "linear":
						pass
					else:
						raise ValueError("Unknown activation for attention: " + str(self.config["attention_activation"]))				
					self.token_probs_all_labels = attention_weights 	

				word_objective_loss = tf.where(tf.sequence_mask(self.sentence_lengths), word_objective_loss, tf.zeros_like(word_objective_loss))
				self.loss += self.config["word_objective_weight"] * tf.reduce_sum(self.word_objective_weights * word_objective_loss)

				attention_weights_argmax = tf.cast(tf.argmax(attention_weights, axis=-1),tf.int32)
				self.prediction_labels = attention_weights_argmax 	
				self.prediction_probs = tf.reduce_max(attention_weights, axis=2) 

				excludeClass0_col_to_zero = [0] 
				excludeClass0_tnsr_shape = tf.shape(attention_weights)
				excludeClass0_attention_weight_mask = [tf.one_hot(col_num*tf.ones((excludeClass0_tnsr_shape[0], excludeClass0_tnsr_shape[1]), dtype=tf.int32), excludeClass0_tnsr_shape[-1]) for col_num in excludeClass0_col_to_zero]
				excludeClass0_attention_weight_mask = tf.reduce_sum(excludeClass0_attention_weight_mask, axis=0)
				excludeClass0_attention_weight_mask = tf.cast(tf.logical_not(tf.cast(excludeClass0_attention_weight_mask, tf.bool)), tf.float32)
				excludeClass0_attention_weights = tf.reduce_max((attention_weights * excludeClass0_attention_weight_mask), axis=2) 	# Max pooling
				self.nonZeroClass_prediction_probs = excludeClass0_attention_weights

				if self.config["masked_attention_weights"] == True:
					entities_to_make_zero = col_to_zero = []					
					unmask1_mask0 = ast.literal_eval(self.config["un_masked_fields_for_attention_weights"])
					print("un_masked_fields_for_attention_weights:",self.config["un_masked_fields_for_attention_weights"])
					print("unmask1_mask0:",unmask1_mask0)
					entities_to_make_zero = [i for i,j in unmask1_mask0.items() if j == 0]
					print("entities_to_make_zero:",entities_to_make_zero)		
					for entity in entities_to_make_zero:
						if entity == 'O':
							col_to_zero.append(0)
						else:
							for tag, idx in self.dict_tags.items():
								if entity == tag:
									col_to_zero.append(idx)						
					tnsr_shape = tf.shape(attention_weights)
					attention_weight_mask = [tf.one_hot(col_num*tf.ones((tnsr_shape[0], tnsr_shape[1]), dtype=tf.int32), tnsr_shape[-1]) for col_num in col_to_zero]
					attention_weight_mask = tf.reduce_sum(attention_weight_mask, axis=0)
					attention_weight_mask = tf.cast(tf.logical_not(tf.cast(attention_weight_mask, tf.bool)), tf.float32)
					masked_attention_weights = tf.reduce_max((attention_weights * attention_weight_mask), axis=2) 	
					if self.config["override_for_aro"] == True:
						self.attention_weights_unnormalised = excludeClass0_attention_weights						
					else:
						self.attention_weights_unnormalised = masked_attention_weights						
				else:
					if self.config["exclude_0_for_aro"] == True:		
						self.attention_weights_unnormalised = excludeClass0_attention_weights
					else:
						self.attention_weights_unnormalised = tf.reduce_max(attention_weights, axis=2)

				if self.config["double_attention_selective_label_weight"] == True:
					selective_attention_weights = tf.where(tf.sequence_mask(self.sentence_lengths), masked_attention_weights, tf.zeros_like(masked_attention_weights))
					selective_attention_weights = selective_attention_weights / tf.reduce_sum(selective_attention_weights, 1, keep_dims=True)   
					processed_tensor = tf.reduce_sum(lstm_outputs * selective_attention_weights[:,:,numpy.newaxis], 1)    
					self.selective_attention_weights_normalized = selective_attention_weights	
				else:
					processed_tensor = None

				if self.config["double_attention_predicted_label_weight"] == True:
					processed_tensor, self.indicative_attention_weights_normalized = self.da_predicted_label_weight(lstm_outputs, processed_tensor)

				if self.config["double_attention_predicted_label"] == True:
					processed_tensor = self.da_predicted_label(attention_weights_argmax,processed_tensor)

				if self.config["double_attention_unsupervised_self_attention"] == True:
					processed_tensor = tf.concat([processed_tensor,unsup_processed_tensor],-1) 					

		if self.config["hidden_layer_size"] > 0:
			processed_tensor = tf.layers.dense(processed_tensor, self.config["hidden_layer_size"], activation=tf.tanh, kernel_initializer=self.initializer)

		self.sentence_scores = tf.layers.dense(processed_tensor, 1, activation=tf.sigmoid, kernel_initializer=self.initializer, name="output_ff")      
		self.sentence_scores = tf.reshape(self.sentence_scores, shape=[tf.shape(processed_tensor)[0]])

		if self.config["sentence_loss_function"] == "crossentropy":		
			sentence_objective_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.sentence_scores, labels=self.sentence_labels)
		else:
			sentence_objective_loss = tf.square(self.sentence_scores - self.sentence_labels)

		self.loss += self.config["sentence_objective_weight"] * tf.reduce_sum(self.sentence_objective_weights * sentence_objective_loss)

		if self.config["attention_objective_weight"] > 0.0:
			self.loss += self.config["attention_objective_weight"] * \
				(tf.reduce_sum(
					self.sentence_objective_weights *tf.square(
						tf.reduce_max(
							tf.where(
								tf.sequence_mask(self.sentence_lengths), 
								self.attention_weights_unnormalised, 
								tf.zeros_like(self.attention_weights_unnormalised) - 1e6), 
							axis=-1) - self.sentence_labels))
				+ 
				tf.reduce_sum(
					self.sentence_objective_weights * tf.square(
						tf.reduce_min(
							tf.where(
								tf.sequence_mask(self.sentence_lengths), 
								self.attention_weights_unnormalised, 
								tf.zeros_like(self.attention_weights_unnormalised) + 1e6), 
							axis=-1) - 0.0)))

		self.token_scores = [tf.where(tf.sequence_mask(self.sentence_lengths), self.prediction_labels, tf.zeros_like(self.prediction_labels))]          
		self.token_probs = [tf.where(tf.sequence_mask(self.sentence_lengths), self.prediction_probs, tf.zeros_like(self.prediction_probs))]    

		if self.config["lmcost_lstm_gamma"] > 0.0:
			self.loss += self.config["lmcost_lstm_gamma"] * self.construct_lmcost(lstm_outputs_fw, lstm_outputs_bw, self.sentence_lengths, self.word_ids, "separate", "lmcost_lstm_separate")
		if self.config["lmcost_joint_lstm_gamma"] > 0.0:
			self.loss += self.config["lmcost_joint_lstm_gamma"] * self.construct_lmcost(lstm_outputs_fw, lstm_outputs_bw, self.sentence_lengths, self.word_ids, "joint", "lmcost_lstm_joint")

		self.train_op = self.construct_optimizer(self.config["opt_strategy"], self.loss, self.learningrate, self.config["clip"])


	def construct_lmcost(self, input_tensor_fw, input_tensor_bw, sentence_lengths, target_ids, lmcost_type, name):
		with tf.variable_scope(name):
			lmcost_max_vocab_size = min(len(self.word2id), self.config["lmcost_max_vocab_size"])
			target_ids = tf.where(tf.greater_equal(target_ids, lmcost_max_vocab_size-1), x=(lmcost_max_vocab_size-1)+tf.zeros_like(target_ids), y=target_ids)
			cost = 0.0
			if lmcost_type == "separate":
				lmcost_fw_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:,1:]
				lmcost_bw_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:,:-1]
				lmcost_fw = self._construct_lmcost(input_tensor_fw[:,:-1,:], lmcost_max_vocab_size, lmcost_fw_mask, target_ids[:,1:], name=name+"_fw")
				lmcost_bw = self._construct_lmcost(input_tensor_bw[:,1:,:], lmcost_max_vocab_size, lmcost_bw_mask, target_ids[:,:-1], name=name+"_bw")
				cost += lmcost_fw + lmcost_bw
			elif lmcost_type == "joint":
				joint_input_tensor = tf.concat([input_tensor_fw[:,:-2,:], input_tensor_bw[:,2:,:]], axis=-1)
				lmcost_mask = tf.sequence_mask(sentence_lengths, maxlen=tf.shape(target_ids)[1])[:,1:-1]
				cost += self._construct_lmcost(joint_input_tensor, lmcost_max_vocab_size, lmcost_mask, target_ids[:,1:-1], name=name+"_joint")
			else:
				raise ValueError("Unknown lmcost_type: " + str(lmcost_type))
			return cost


	def _construct_lmcost(self, input_tensor, lmcost_max_vocab_size, lmcost_mask, target_ids, name):
		with tf.variable_scope(name):
			lmcost_hidden_layer = tf.layers.dense(input_tensor, self.config["lmcost_hidden_layer_size"], activation=tf.tanh, kernel_initializer=self.initializer)
			lmcost_output = tf.layers.dense(lmcost_hidden_layer, lmcost_max_vocab_size, activation=None, kernel_initializer=self.initializer)
			lmcost_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lmcost_output, labels=target_ids)
			lmcost_loss = tf.where(lmcost_mask, lmcost_loss, tf.zeros_like(lmcost_loss))
			return tf.reduce_sum(lmcost_loss)


	def construct_optimizer(self, opt_strategy, loss, learningrate, clip):
		optimizer = None
		if opt_strategy == "adadelta":
			optimizer = tf.train.AdadeltaOptimizer(learning_rate=learningrate)
		elif opt_strategy == "adam":
			optimizer = tf.train.AdamOptimizer(learning_rate=learningrate)
		elif opt_strategy == "sgd":
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)
		else:
			raise ValueError("Unknown optimisation strategy: " + str(opt_strategy))

		if clip > 0.0:
			grads, vs = zip(*optimizer.compute_gradients(loss))
			grads, gnorm  = tf.clip_by_global_norm(grads, clip)
			train_op = optimizer.apply_gradients(zip(grads, vs))
		else:
			train_op = optimizer.minimize(loss)
		return train_op


	def preload_word_embeddings(self, embedding_path):
		loaded_embeddings = set()
		embedding_matrix = self.session.run(self.word_embeddings)
		with open(embedding_path, 'r') as f:
			for line in f:
				line_parts = line.strip().split()
				if len(line_parts) <= 2:
					continue
				w = line_parts[0]
				if self.config["lowercase"] == True:
					w = w.lower()
				if self.config["replace_digits"] == True:
					w = re.sub(r'\d', '0', w)
				if w in self.word2id and w not in loaded_embeddings:
					word_id = self.word2id[w]
					embedding = numpy.array(line_parts[1:])
					embedding_matrix[word_id] = embedding
					loaded_embeddings.add(w)
		self.session.run(self.word_embeddings.assign(embedding_matrix))
		print("n_preloaded_embeddings: " + str(len(loaded_embeddings)))


	def translate2id(self, token, token2id, unk_token, lowercase=False, replace_digits=False, singletons=None, singletons_prob=0.0):
		if lowercase == True:
			token = token.lower()
		if replace_digits == True:
			token = re.sub(r'\d', '0', token)

		token_id = None
		if singletons != None and token in singletons and token in token2id and unk_token != None and numpy.random.uniform() < singletons_prob:
			token_id = token2id[unk_token]
		elif token in token2id:
			token_id = token2id[token]
		elif unk_token != None:
			token_id = token2id[unk_token]
		else:
			raise ValueError("Unable to handle value, no UNK token: " + str(token))
		return token_id


	def create_input_dictionary_for_batch(self, batch, is_training, learningrate):
		sentence_lengths = numpy.array([len(sentence) for sentence in batch])
		max_sentence_length = sentence_lengths.max()
		max_word_length = numpy.array([numpy.array([len(word[0]) for word in sentence]).max() for sentence in batch]).max()
		if self.config["allowed_word_length"] > 0 and self.config["allowed_word_length"] < max_word_length:
			max_word_length = min(max_word_length, self.config["allowed_word_length"])

		word_ids = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.int32)
		char_ids = numpy.zeros((len(batch), max_sentence_length, max_word_length), dtype=numpy.int32)
		word_lengths = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.int32)
		word_labels = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.int32)            
		word_labels_oh = numpy.zeros((len(batch), max_sentence_length, self.config["ntags"]), dtype=numpy.float32)        
		sentence_labels = numpy.zeros((len(batch)), dtype=numpy.float32)
		word_objective_weights = numpy.zeros((len(batch), max_sentence_length), dtype=numpy.float32)
		sentence_objective_weights = numpy.zeros((len(batch)), dtype=numpy.float32)

		if is_training == False:
			self.dict_tags = dict()
			with open(self.config["filename_tags"]) as f:
				for idx, word in enumerate(f):
					word = word.strip()
					self.dict_tags[word] = idx+1      
		
		singletons = self.singletons if is_training == True else None
		singletons_prob = self.config["singletons_prob"] if is_training == True else 0.0
		for i in range(len(batch)):
			count_interesting_labels = numpy.array([1.0 if batch[i][j][-1] in ['B-ADE', 'I-ADE', 'ADE'] else 0.0 for j in range(len(batch[i]))]).sum()
			sentence_labels[i] = 1.0 if count_interesting_labels >= 1.0 else 0.0            

			for j in range(len(batch[i])):
				word_ids[i][j] = self.translate2id(batch[i][j][0], self.word2id, self.UNK, lowercase=self.config["lowercase"], replace_digits=self.config["replace_digits"], singletons=singletons, singletons_prob=singletons_prob)

				if batch[i][j][-1] == self.config["default_label"]:
					word_labels[i][j] = 0
					word_labels_oh[i][j][0] = 1
				else:                
					for tag, idx in self.dict_tags.items():
						if batch[i][j][-1] == tag:
							word_labels[i][j] = idx                                   
							word_labels_oh[i][j][idx] = 1                

				word_lengths[i][j] = len(batch[i][j][0])
				for k in range(min(len(batch[i][j][0]), max_word_length)):
					char_ids[i][j][k] = self.translate2id(batch[i][j][0][k], self.char2id, self.CUNK)

				if len(batch[i][j]) == 2 or (len(batch[i][j]) >= 3 and batch[i][j][1] == "T"):
					word_objective_weights[i][j] = 1.0  

			if len(batch[i][j]) == 2 or (len(batch[i][j]) >= 3 and batch[i][0][1] == "S") or self.config["sentence_objective_persistent"] == True:
				sentence_objective_weights[i] = 1.0

		input_dictionary = {self.word_ids: word_ids, self.char_ids: char_ids, self.sentence_lengths: sentence_lengths, self.word_lengths: word_lengths, self.word_labels: word_labels, self.word_labels_oh: word_labels_oh, self.word_objective_weights: word_objective_weights, self.sentence_labels: sentence_labels, self.sentence_objective_weights: sentence_objective_weights, self.learningrate: learningrate, self.is_training: is_training}
		return input_dictionary


	def process_batch(self, batch, is_training, learningrate):
		feed_dict = self.create_input_dictionary_for_batch(batch, is_training, learningrate)
		cost, sentence_scores, token_scores, token_probs, token_probs_all_labels, unsup_weights, selective_weights, indicative_weights = self.session.run([self.loss, self.sentence_scores, self.token_scores, self.token_probs, self.token_probs_all_labels, self.unsup_attention_weights_normalised, self.selective_attention_weights_normalized, self.indicative_attention_weights_normalized] + ([self.train_op] if is_training == True else []), feed_dict=feed_dict)[:8] 
		return cost, sentence_scores, token_scores, token_probs, token_probs_all_labels, unsup_weights, selective_weights, indicative_weights

	def initialize_session(self):
		tf.set_random_seed(self.config["random_seed"])
		session_config = tf.ConfigProto()
		session_config.gpu_options.allow_growth = self.config["tf_allow_growth"]
		session_config.gpu_options.per_process_gpu_memory_fraction = self.config["tf_per_process_gpu_memory_fraction"]
		self.session = tf.Session(config=session_config)
		self.session.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=1)


	def get_parameter_count(self):
		total_parameters = 0
		for variable in tf.trainable_variables():
			shape = variable.get_shape()
			variable_parameters = 1
			for dim in shape:
				variable_parameters *= dim.value
			total_parameters += variable_parameters
		return total_parameters


	def get_parameter_count_without_word_embeddings(self):
		shape = self.word_embeddings.get_shape()
		variable_parameters = 1
		for dim in shape:
			variable_parameters *= dim.value
		return self.get_parameter_count() - variable_parameters


	def save(self, filename):
		dump = {}
		dump["config"] = self.config
		dump["UNK"] = self.UNK
		dump["CUNK"] = self.CUNK
		dump["word2id"] = self.word2id
		dump["char2id"] = self.char2id
		dump["singletons"] = self.singletons

		dump["params"] = {}
		for variable in tf.global_variables():
			assert(variable.name not in dump["params"]), "Error: variable with this name already exists" + str(variable.name)
			dump["params"][variable.name] = self.session.run(variable)
		with open(filename, 'wb') as f:
			pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)


	@staticmethod
	def load(filename, new_config=None):
		with open(filename, 'rb') as f:
			dump = pickle.load(f)

			# for safety, so we don't overwrite old models
			dump["config"]["save"] = None

			# we use the saved config, except for values that are present in the new config
			if new_config != None:
				for key in new_config:
					dump["config"][key] = new_config[key]

			labeler = MLTModel(dump["config"])
			labeler.UNK = dump["UNK"]
			labeler.CUNK = dump["CUNK"]
			labeler.word2id = dump["word2id"]
			labeler.char2id = dump["char2id"]
			labeler.singletons = dump["singletons"]

			labeler.construct_network()
			labeler.initialize_session()

			labeler.load_params(filename)

			return labeler


	def load_params(self, filename):
		with open(filename, 'rb') as f:
			dump = pickle.load(f)

			for variable in tf.global_variables():
				assert(variable.name in dump["params"]), "Variable not in dump: " + str(variable.name)
				assert(variable.shape == dump["params"][variable.name].shape), "Variable shape not as expected: " + str(variable.name) + " " + str(variable.shape) + " " + str(dump["params"][variable.name].shape)
				value = numpy.asarray(dump["params"][variable.name])
				self.session.run(variable.assign(value))
