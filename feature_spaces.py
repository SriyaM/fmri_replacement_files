import os
import sys
import numpy as np
import json
from os.path import join, dirname
from tqdm import tqdm
import pickle
from huggingface_hub import login


from ridge_utils.interpdata import lanczosinterp2D
from ridge_utils.SemanticModel import SemanticModel
from ridge_utils.dsutils import *
from ridge_utils.stimulus_utils import load_textgrids, load_simulated_trfiles
from ridge_utils.stimulus_utils import load_textgrids, load_simulated_trfiles

from config import REPO_DIR, EM_DATA_DIR, DATA_DIR

import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel, BertForMaskedLM, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# from mech_lm.experiments.inference_neuro import get_infini_embeddings,  setup_model, generate_embeddings
from mech_lm2.experiments.inference_neuro import setup_model, generate_embeddings

def get_save_location(primary_dir, fallback_dir, feature, numeric_mod, subject):
	primary_location = join(primary_dir, "results", feature + "_" + numeric_mod, subject)
	fallback_location = join(fallback_dir, "results", feature + "_" + numeric_mod, subject)

	try:
		# Try to create the directory in the primary location
		os.makedirs(primary_location, exist_ok=True)
		return primary_location
	except (OSError, IOError) as e:
		# If an error occurs, use the fallback location
		os.makedirs(fallback_location, exist_ok=True)
		return fallback_location

def get_story_wordseqs(stories):
	grids = load_textgrids(stories, DATA_DIR)
	with open(join(DATA_DIR, "ds003020/derivative/respdict.json"), "r") as f:
		respdict = json.load(f)
	# a dictionary of words listened to between the start and end time
	trfiles = load_simulated_trfiles(respdict)
	# returns a dictionary with dataseq stories that have a transcript with filtered words and metadata
	wordseqs = make_word_ds(grids, trfiles)
	return wordseqs

def get_story_wordseqs_no_filter(stories):
	grids = load_textgrids(stories, DATA_DIR)
	with open(join(DATA_DIR, "ds003020/derivative/respdict.json"), "r") as f:
		respdict = json.load(f)
	# a dictionary of words listened to between the start and end time
	trfiles = load_simulated_trfiles(respdict)
	# returns a dictionary with dataseq stories that have a transcript with filtered words and metadata
	wordseqs = make_word_ds_no_filter(grids, trfiles)
	return wordseqs


def get_story_wordseqs_filter(stories, include):
	grids = load_textgrids(stories, DATA_DIR)
	with open(join(DATA_DIR, "ds003020/derivative/respdict.json"), "r") as f:
		respdict = json.load(f)
	# a dictionary of words listened to between the start and end time
	trfiles = load_simulated_trfiles(respdict)
	# returns a dictionary with dataseq stories that have a transcript with filtered words and metadata
	wordseqs = make_word_ds_filter(grids, trfiles, include)
	return wordseqs

def get_story_wordseqs_filter_non_ind(stories, test_stories, train_stories, rem):
	grids = load_textgrids(stories, DATA_DIR)
	with open(join(DATA_DIR, "ds003020/derivative/respdict.json"), "r") as f:
		respdict = json.load(f)
	# a dictionary of words listened to between the start and end time
	trfiles = load_simulated_trfiles(respdict)
	# returns a dictionary with dataseq stories that have a transcript with filtered words and metadata
	wordseqs = make_word_ds_non_ind_filter(grids, trfiles, test_stories, train_stories, rem)
	return wordseqs


def get_story_phonseqs(stories):
	grids = load_textgrids(stories, DATA_DIR)
	with open(join(DATA_DIR, "ds003020/derivative/respdict.json"), "r") as f:
		respdict = json.load(f)
	trfiles = load_simulated_trfiles(respdict)
	wordseqs = make_phoneme_ds(grids, trfiles)
	return wordseqs

def downsample_word_vectors(stories, word_vectors, wordseqs):
	"""Get Lanczos downsampled word_vectors for specified stories.

	Args:
		stories: List of stories to obtain vectors for.
		word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	downsampled_semanticseqs = dict()
	for story in stories:
		downsampled_semanticseqs[story] = lanczosinterp2D(
			word_vectors[story], wordseqs[story].data_times, 
			wordseqs[story].tr_times, window=3)
	return downsampled_semanticseqs

###########################################
########## ARTICULATORY Features ##########
###########################################

def ph_to_articulate(ds, ph_2_art):
	""" Following make_phoneme_ds converts the phoneme DataSequence object to an 
	articulate Datasequence for each grid.
	"""
	articulate_ds = []
	for ph in ds:
		try:
			articulate_ds.append(ph_2_art[ph])
		except:
			articulate_ds.append([""])
	return articulate_ds

articulates = ["bilabial","postalveolar","alveolar","dental","labiodental",
			   "velar","glottal","palatal", "plosive","affricative","fricative",
			   "nasal","lateral","approximant","voiced","unvoiced","low", "mid",
			   "high","front","central","back"]

def histogram_articulates(ds, data, articulateset=articulates):
	"""Histograms the articulates in the DataSequence [ds]."""
	final_data = []
	for art in ds:
		final_data.append(np.isin(articulateset, art))
	final_data = np.array(final_data)
	return (final_data, data.split_inds, data.data_times, data.tr_times)

def get_articulation_vectors(allstories):
	"""Get downsampled articulation vectors for specified stories.
	Args:
		allstories: List of stories to obtain vectors for.
	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	with open(join(EM_DATA_DIR, "articulationdict.json"), "r") as f:
		artdict = json.load(f)
	phonseqs = get_story_phonseqs(allstories) #(phonemes, phoneme_times, tr_times)
	downsampled_arthistseqs = {}
	for story in allstories:
		olddata = np.array(
			[ph.upper().strip("0123456789") for ph in phonseqs[story].data])
		ph_2_art = ph_to_articulate(olddata, artdict)
		arthistseq = histogram_articulates(ph_2_art, phonseqs[story])
		downsampled_arthistseqs[story] = lanczosinterp2D(
			arthistseq[0], arthistseq[2], arthistseq[3])
	return downsampled_arthistseqs

###########################################
########## PHONEME RATE Features ##########
###########################################

def get_phonemerate_vectors(allstories):
	"""Get downsampled phonemerate vectors for specified stories.
	Args:
		allstories: List of stories to obtain vectors for.
	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	with open(join(EM_DATA_DIR, "articulationdict.json"), "r") as f:
		artdict = json.load(f)
	phonseqs = get_story_phonseqs(allstories) #(phonemes, phoneme_times, tr_times)
	downsampled_arthistseqs = {}
	for story in allstories:
		olddata = np.array(
			[ph.upper().strip("0123456789") for ph in phonseqs[story].data])
		ph_2_art = ph_to_articulate(olddata, artdict)
		arthistseq = histogram_articulates(ph_2_art, phonseqs[story])
		nphonemes = arthistseq[0].shape[0]
		phonemerate = np.ones([nphonemes, 1])
		downsampled_arthistseqs[story] = lanczosinterp2D(
			phonemerate, arthistseq[2], arthistseq[3])
	return downsampled_arthistseqs


###########################################
########## PERPLEXITY Features ############
###########################################

def get_perplexity_features(allstories):
	"""Get downsampled phonemerate vectors for specified stories.
	Args:
		allstories: List of stories to obtain vectors for.
	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	# Iterate through the split indices and run one for each
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	model = GPT2LMHeadModel.from_pretrained('gpt2')
	
	# Move model to GPU if available
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.to(device)
	model.eval()

	print("LOADED")
	
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}

	window_size = 5
	
	for story in allstories:
		perplexities = []
		current = wordseqs[story]
		story_words = current.data

		for i, word in enumerate(story_words):
			start = max(0, i - window_size + 1)
			end = i + 1
			text_segment = story_words[start:end]

			text = ' '.join(text_segment)
			
			if len(text) > 0:

				inputs = tokenizer(text, return_tensors='pt').to(device)
				input_ids = inputs['input_ids']

				with torch.no_grad():
					outputs = model(input_ids, labels=input_ids)
					log_likelihoods = outputs.loss
				
				# Clamp log likelihoods to avoid numerical instability
				average_negative_log_likelihood = torch.clamp(log_likelihoods, min=-100, max=100).item()
				
				 # Check for NaN values
				if np.isnan(average_negative_log_likelihood):
					#print(f"NaN detected in log likelihoods for story {story} at index {i}")
					perplexity = 0 # or use another sentinel value like -1
				else:
					perplexity = np.exp(average_negative_log_likelihood)
				
				"""if perplexity != 0:
					perplexity = math.log(perplexity)"""
				
				print(perplexity)

				perplexities.append([perplexity])
			
			else:

				perplexities.append([0])

		vectors[story] = np.array(perplexities)
	
	return downsample_word_vectors(allstories, vectors, wordseqs)


######################################################
########## ENG1000 Features With Perplexity ##########
######################################################

def get_eng1000_and_perplexity(allstories, k):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	# Iterate through the split indices and run one for each
	tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
	model = GPT2LMHeadModel.from_pretrained('gpt2')

	
	# Move model to GPU if available
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model.to(device)
	model.eval()

	print("LOADED")
	
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		
		perplexities = []
		current = wordseqs[story]
		story_words = current.data

		combined = []

		window_size = k

		for i, w in enumerate(story_words):
			start = max(0, i - window_size + 1)
			end = i + 1
			text_segment = story_words[start:end]

			text = ' '.join(text_segment)
			
			if len(text) > 0:

				inputs = tokenizer(text, return_tensors='pt').to(device)
				input_ids = inputs['input_ids']

				with torch.no_grad():
					outputs = model(input_ids, labels=input_ids)
					log_likelihoods = outputs.loss
				
				# Clamp log likelihoods to avoid numerical instability
				average_negative_log_likelihood = torch.clamp(log_likelihoods, min=-100, max=100).item()
				
				# Check for NaN values
				if np.isnan(average_negative_log_likelihood):
					print(f"NaN detected in log likelihoods for story {story} at index {i}")
					perplexity = 0 # or use another sentinel value like -1
				else:
					perplexity = np.exp(average_negative_log_likelihood)
				
				perplexities.append([perplexity])
			else:

				perplexities.append([0])

		for i in range(len(sm.data)):
			combined.append(np.append(sm.data[i], perplexities[i]))

		vectors[story] = combined

	return downsample_word_vectors(allstories, vectors, wordseqs)

########################################
########## WORD RATE Features ##########
########################################

def get_wordrate_vectors(allstories):
	"""Get wordrate vectors for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	for story in allstories:
		nwords = len(wordseqs[story].data)
		vectors[story] = np.ones([nwords, 1])
	
	return downsample_word_vectors(allstories, vectors, wordseqs)


######################################
########## ENG1000 Features ##########
######################################

def get_eng1000_vectors(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}

	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		vectors[story] = sm.data
	
	for story in allstories:
		print("LENGTHS", "time", story, len(wordseqs[story].data), len(wordseqs[story].data_times), len(vectors[story]), len(wordseqs[story].split_inds))
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng1000_ind_only(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	inc_ind = {}

	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		# The last argument is the time window
		vectors[story], inc_ind[story] = induction_head_eng_filter(wordseqs[story], word_vectors, 985, x, k)
	
	wordseqs_f = get_story_wordseqs_filter(allstories, inc_ind)
		
	return downsample_word_vectors(allstories, vectors, wordseqs_f), inc_ind

def get_eng1000_non_ind_only(allstories, test_stories, train_stories, rem, x, k):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}

	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		vectors[story] = sm.data
		if story in test_stories:
			vectors[story] = eng_non_ind_filter(wordseqs[story], vectors[story], train_stories, test_stories, rem[story], 985)
	
	wordseqs_f = get_story_wordseqs_filter_non_ind(allstories, test_stories, train_stories, rem)

	# for story in allstories:
	# 	print("LENGTHS", "time", story, len(wordseqs_f[story].data), len(wordseqs_f[story].data_times), len(vectors[story]), len(wordseqs_f[story].tr_times), len(rem[story]))

	return downsample_word_vectors(allstories, vectors, wordseqs_f)


def get_eng1000_w_induction_zero(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_head_zero(wordseqs[story], word_vectors, 985, x)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng1000_w_induction_zero_b(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_head_zero_b(wordseqs[story], word_vectors, 985, x)
	return downsample_word_vectors(allstories, vectors, wordseqs)


def get_eng1000_w_both_induction_zero(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = fuzzy_induction_both_weight_zero(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng1000_w_both_induction_zero_b(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = fuzzy_induction_both_weight_zero(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)


######################################
########## ENG1000 Features ##########
######################################

def get_eng1000_vectors_2x(allstories):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		vectors[story] = np.concatenate([sm.data, sm.data], axis=1)
	return downsample_word_vectors(allstories, vectors, wordseqs)
	
#############################################################
########## ENG1000 w/ Fuzzy Next Word Distribution LLama ##########
#############################################################

def get_eng1000_vectors_w_fuzzy_distribution_llama(allstories, x):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""

    # Use AutoTokenizer to automatically select the correct tokenizer class   

	os.environ['HUGGINGFACE_TOKEN'] = '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'
	model_name = 'meta-llama/Meta-Llama-3-8B'
	token = os.getenv('HUGGINGFACE_TOKEN')
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
	model = AutoModel.from_pretrained(model_name, use_auth_token=token).to('cuda')

	model.to('cuda')  # Move the model to GPU
	model.eval()

	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		print(story)
		vectors[story] = induction_fuzzy_distribution_llama(wordseqs[story], word_vectors, 985, 1, x, tokenizer, model)
	return downsample_word_vectors(allstories, vectors, wordseqs)

#############################################################
########## ENG1000 w/ Fuzzy Next Word Distribution ##########
#############################################################

def get_eng1000_vectors_w_fuzzy_distribution(allstories):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertForMaskedLM.from_pretrained('bert-base-uncased')
	model.to('cuda')  # Move the model to GPU
	model.eval()

	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_fuzzy_distribution(wordseqs[story], word_vectors, 985, 1, tokenizer, model)
	return downsample_word_vectors(allstories, vectors, wordseqs)

#####################################################
########## ENG1000 Features With Induction ##########
#####################################################

def get_eng1000_vectors_w_induction(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		# The last argument is the time window
		vectors[story] = induction_head(wordseqs[story], word_vectors, 985, x)
	return downsample_word_vectors(allstories, vectors, wordseqs)


##############################################################
########## ENG1000 Features With Weighted Induction ##########
##############################################################

def get_eng1000_vectors_w_weighted_induction(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		# The last argument is the time window
		vectors[story] = induction_head_weighted(wordseqs[story], word_vectors, 985, x)
	return downsample_word_vectors(allstories, vectors, wordseqs)

#################################################################
########## ENG1000 Features With GPT Pred Induction ###########
#################################################################

def get_eng1000_vectors_w_pred_induction(allstories):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		print(story)
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_head_pred(wordseqs[story], word_vectors, 985, eng1000)
	return downsample_word_vectors(allstories, vectors, wordseqs)

###################################################
########## ENG1000 with Random Induction ##########
###################################################


def get_eng1000_w_random_induction_k(allstories, k):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""

	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		print(sm.data_times[-2])
		# Last argument is the number of seconds to go back and test for
		vectors[story] = induction_head_random_k(wordseqs[story], word_vectors, 985, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng1000_w_random_induction(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""

	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		print(sm.data_times[-2])
		# Last argument is the number of seconds to go back and test for
		vectors[story] = induction_head_random(wordseqs[story], word_vectors, 985, x)
	return downsample_word_vectors(allstories, vectors, wordseqs)


###################################################
########## ENG1000 with Induction Avg Next ##########
###################################################

def get_eng1000_w_avg_next_induction(allstories):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""

	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_head_avg_next_same_word(wordseqs[story], word_vectors, 985, 2)
	return downsample_word_vectors(allstories, vectors, wordseqs)

###################################################
########## ENG1000 with Induction Avg Prev ##########
###################################################

def get_eng1000_w_avg_prev_induction(allstories):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""

	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_head_avg_prev_same_word(wordseqs[story], word_vectors, 985, 4)
	return downsample_word_vectors(allstories, vectors, wordseqs)

###################################################
########## ENG1000 with Fuzzy Induction ###########
###################################################

def get_eng1000_w_fuzzy_induction(allstories, x):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""

	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = fuzzy_induction_dist_weight(wordseqs[story], word_vectors, 985, x, 1)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng1000_w_fuzzy_induction_sd(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""

	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = fuzzy_induction_both_weight(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng1000_w_fuzzy_induction_s(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""

	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = fuzzy_induction_sim_weight(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng1000_w_fuzzy_induction_d(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""

	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = fuzzy_induction_dist_weight(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng1000_w_fuzzy_induction_a(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""

	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = fuzzy_induction_avg_weight(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)


######################################
########## BERT Features #############
######################################

def create_input(text, tokenizer, max_length):
  stokens = tokenizer.tokenize(text)
  stokens = ["[CLS]"] + stokens + ["[SEP]"]
  input_ids = get_ids(tokenizer,stokens, max_length)
  input_masks = get_masks(stokens, max_length)
  input_ids = torch.tensor([input_ids]).to('cuda')  # Move input tensors to GPU
  input_masks = torch.tensor([input_masks]).to('cuda')  # Move input tensors to GPU
  return stokens, input_ids, input_masks

def get_ids(tokenizer, token, max_length):
  token_ids = tokenizer.convert_tokens_to_ids(token)
  input_ids = token_ids + [0]*(max_length - len(token_ids))
  return input_ids

def get_masks(tokens, max_length):
  if len(tokens)>max_length:
     raise IndexError("Token length greater than max length")
  return [1]*len(tokens) + [0]*(max_length - len(tokens))

def get_bert_embeddings(tokens_tensor, masked_tensors, model):
   with torch.no_grad():
      outputs = model(tokens_tensor, attention_mask=masked_tensors)
      hidden_states = outputs.hidden_states
      token_embeddings = hidden_states[-1]
   token_embeddings = torch.squeeze(token_embeddings, dim=0)
   list_token_embeddings = [token_embed.tolist() for token_embed in  token_embeddings]
   return list_token_embeddings

def get_bert_vectors(allstories):
	
	tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
	model = BertModel.from_pretrained("bert-large-uncased", output_hidden_states=True)
	model = model.to('cuda')  # Move model to GPU
	print("LOADED")

	words_per_chunk = 32
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}

	for story in allstories:
		print("story name: ", story)
		story_vectors = []
		story_seq = wordseqs[story]
		
		num_words = len(story_seq.data)

		for i in tqdm(range(num_words)):
			
			# Calculate the start and end indices for the chunk
			start = max(0, i - words_per_chunk + 1)
			chunk = story_seq.data[start : i + 1]
			text = " ".join(chunk)
			if len(text) > 500:
				print(text)
				print(i, start)

			stokens, input_ids, input_masks = create_input(text, tokenizer, 512)
			chunk_emb = get_bert_embeddings(input_ids, input_masks, model)
			
			word_emb = chunk_emb[-2]
			story_vectors.append(word_emb)

		vectors[story] = story_vector
	
	return downsample_word_vectors(allstories, vectors, wordseqs)

######################################
# ENG1000 Double Window Induction

def get_eng_window_vectors(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_head_eng_window(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng_window_avg(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_head_eng_window_avg(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng_k_window_avg(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_head_eng_k_window_avg(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng_k_window(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_head_eng_k_window(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng_window_vectors_b(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_head_eng_window_baseline(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)


def get_eng_window_avg_b(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_head_eng_window_avg_baseline(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng_k_window_avg_b(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_head_eng_k_window_avg_baseline(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_eng_k_window_b(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		word_vectors = sm.data
		vectors[story] = induction_head_eng_k_window_baseline(wordseqs[story], word_vectors, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

######################################
########## Llama Features ############
######################################

def get_llama_w_tr_induction(allstories, x, k, subject, model=None, tokenizer=None):
	"""Get llama vectors with induction vectors for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	vec_location_suffix = "incont_infinigram"
	top_percent = k

	vectors = {}
	resp = {}
	vec_location = DATA_DIR[:-5] + f"/infini_pca_emb{x}/{vec_location_suffix}"
	resp_location = DATA_DIR[:-5] + "/resp_pca/100"

	if model==None and tokenizer==None:
		os.environ['HUGGINGFACE_TOKEN'] = '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'
		token = os.getenv('HUGGINGFACE_TOKEN')

		# Load the tokenizer and model with authentication token
		tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', use_auth_token=token)
		model = AutoModel.from_pretrained('meta-llama/Meta-Llama-3-8B', use_auth_token=token)
		model.eval()

		# Check if ROCm is available and move model to GPU
		if torch.cuda.is_available():  # ROCm uses the same torch.cuda API
			device = torch.device('cuda')
			print(f"Using GPU: {torch.cuda.get_device_name(0)}")
			sys.stdout.flush()
		else:
			device = torch.device('cpu')
			print("ROCm not available, using CPU.")
			sys.stdout.flush()

		model.to(device)
		
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding((wordseqs[story]).data, 16, model, tokenizer)
		print(word_vectors.shape)
		print( "num_story_wrods" , len(wordseqs[story].data))
		vectors[story] =  word_vectors

	tr_vectors = {}
	for story in allstories:
		# Load Infinigram Vectors
		cur_vec_file = os.path.join(vec_location, story, "vectors.pkl")
		with open(cur_vec_file, 'rb') as file:
			infini_vectors = pickle.load(file)

		# Load PCA responses
		pca_file = os.path.join(resp_location, story, "vectors.pkl")
		with open(pca_file, 'rb') as file:
			resp_vectors = pickle.load(file)
			resp[story] = resp_vectors

		llama_vectors = vectors[story]
		fin_vectors = pca_tr_infini_prep(wordseqs[story], llama_vectors, infini_vectors, resp_vectors)
		tr_vectors[story] = fin_vectors

	print("done with initial vector creation")

	# Downsample Vectors
	downsampled = downsample_word_vectors(allstories, tr_vectors, wordseqs)
	# Adjust them in the correct way
	new_features = dict()
	for story in allstories:
		new_vector = interp_induction_llama(wordseqs[story], llama_vectors, infini_vectors, resp[story], downsampled[story], top_percent)
		# Validate the dimension
		print("dimension of new TR", story, new_vector.shape)
		new_features[story] = new_vector

	## CODE IS DONE FOR THE VECTORS
	return new_features




def get_llama_ind_only(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	inc_ind = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding((wordseqs[story]).data, k)
		vectors[story], inc_ind[story] = induction_head_llama_filtered(wordseqs[story], word_vectors, 4096, x, k)
	
	wordseqs_f = get_story_wordseqs_filter(allstories, inc_ind)
	return downsample_word_vectors(allstories, vectors, wordseqs_f), inc_ind

def get_llama_non_ind_only(allstories, test_stories, train_stories, rem, x, k):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	# The function will cut out relevant chunks in the test stories embeddings
	# The function will cut out relevant chunks in the wordseq of test stories
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		vectors[story] = get_last_token_embedding((wordseqs[story]).data, k)
		if story in test_stories:
			vectors[story] = llama_non_ind_filter(wordseqs[story], vectors[story], train_stories, test_stories, rem[story], 4096)
	
	wordseqs_f = get_story_wordseqs_filter_non_ind(allstories, test_stories, train_stories, rem)

	for story in allstories:
		print("LENGTHS", "time", story, len(wordseqs_f[story].data), len(wordseqs_f[story].data_times), len(vectors[story]), len(wordseqs_f[story].tr_times), len(rem[story]))

	return downsample_word_vectors(allstories, vectors, wordseqs_f)

def get_llama_vectors(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding((wordseqs[story]).data, k)
		vectors[story] = induction_head_llama(wordseqs[story], word_vectors, 4096, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_llama_2x(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding((wordseqs[story]).data, k)
		print( "num_story_wrods" , len(wordseqs[story].data))
		print(word_vectors.shape)
		vectors[story] =  np.concatenate([word_vectors, word_vectors], axis=1)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_llama_window_avg(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding((wordseqs[story]).data, k)
		vectors[story] = induction_head_llama_avg(wordseqs[story], word_vectors, 4096, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_llama(allstories, x, k, subject, model=None, tokenizer=None):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	if model==None and tokenizer==None:
		os.environ['HUGGINGFACE_TOKEN'] = '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'
		token = os.getenv('HUGGINGFACE_TOKEN')

		# Load the tokenizer and model with authentication token
		tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', use_auth_token=token)
		model = AutoModel.from_pretrained('meta-llama/Meta-Llama-3-8B', use_auth_token=token)
		model.eval()

		# Check if ROCm is available and move model to GPU
		if torch.cuda.is_available():  # ROCm uses the same torch.cuda API
			device = torch.device('cuda')
			print(f"Using GPU: {torch.cuda.get_device_name(0)}")
			sys.stdout.flush()
		else:
			device = torch.device('cpu')
			print("ROCm not available, using CPU.")
			sys.stdout.flush()

		model.to(device)
		
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding((wordseqs[story]).data, k, model, tokenizer)
		print(word_vectors.shape)
		print( "num_story_wrods" , len(wordseqs[story].data))
		vectors[story] =  word_vectors
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_llama_k_window_avg(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding((wordseqs[story]).data, k)
		vectors[story] = induction_head_llama_k_avg(wordseqs[story], word_vectors, 4096, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_llama_k_window(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding((wordseqs[story]).data, k)
		vectors[story] = induction_head_llama_k(wordseqs[story], word_vectors, 4096, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

""""
def get_llama_window_eng(allstories, x, k, subject):

	wordseqs = get_story_wordseqs(allstories)
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding((wordseqs[story]).data, k)
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		vectors[story] = induction_head_llama_eng(wordseqs[story], word_vectors, sm.data, 985, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)"""

def get_llama_window_avg_b(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding((wordseqs[story]).data, k)
		vectors[story] = induction_head_llama_avg_baseline(wordseqs[story], word_vectors, 4096, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_llama_k_window_avg_b(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding((wordseqs[story]).data, k)
		vectors[story] = induction_head_llama_k_avg_baseline(wordseqs[story], word_vectors, 4096, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_llama_k_window_b(allstories, x, k, subject):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding((wordseqs[story]).data, k)
		vectors[story] = induction_head_llama_k_baseline(wordseqs[story], word_vectors, 4096, x, k)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_last_token_embedding(words, window_size, model, tokenizer,  batch_size=64):

	# Check if ROCm is available and move model to GPU
	if torch.cuda.is_available():  # ROCm uses the same torch.cuda API
		device = torch.device('cuda')
		print(f"Using GPU: {torch.cuda.get_device_name(0)}")
		sys.stdout.flush()
	else:
		device = torch.device('cpu')
		print("ROCm not available, using CPU.")
		sys.stdout.flush()

	# Manually set padding token if it's not set
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	# Record indices of empty words and remove them from the list
	empty_word_indices = [idx for idx, word in enumerate(words) if not word]
	trimmed_words = [word for word in words if word]

	# Tokenize the entire text once
	encoded_input = tokenizer(trimmed_words, return_tensors='pt', padding=True, truncation=False, is_split_into_words=True)
	input_ids = encoded_input['input_ids'][0]
	attention_mask = encoded_input['attention_mask'][0]

	# Move to device
	input_ids = input_ids.to(device)
	attention_mask = attention_mask.to(device)

	#print(trimmed_words)
	#print("inp_id_len", len(input_ids))

	# Map word indices to token indices
	word_to_token_indices = []
	current_token_idx = 0
	for word in trimmed_words:
		tokenized_word = tokenizer.tokenize(word)
		token_indices = list(range(current_token_idx, current_token_idx + len(tokenized_word)))
		word_to_token_indices.append(token_indices)
		current_token_idx += len(tokenized_word)

	all_embeddings = []

	# Prepare context windows based on word indices
	context_indices = []
	for idx, token_indices in enumerate(word_to_token_indices):
		start_idx = max(0, idx - window_size + 1)
		end_idx = idx + 1
		context_window_tokens = [tok for word_idx in range(start_idx, end_idx) for tok in word_to_token_indices[word_idx]]
		context_indices.append((context_window_tokens, token_indices[-1]))

	# Process context windows in batches
	for i in tqdm(range(0, len(context_indices), batch_size)):
		batch_contexts = context_indices[i:i + batch_size]

		# Ensure no out-of-bounds tokens before creating input batches
		valid_batch_contexts = []
		for tokens, token_idx in batch_contexts:
			if all(0 <= t < len(input_ids) for t in tokens):
				valid_batch_contexts.append((tokens, token_idx))
			else:
				print(f"Skipping out-of-bounds context: tokens={tokens}, token_idx={token_idx}")
				print(len(input_ids))

		if not valid_batch_contexts:
			continue

		# Create input batches
		input_ids_batch = torch.nn.utils.rnn.pad_sequence([input_ids[torch.tensor(tokens)].to(device) for tokens, _ in valid_batch_contexts], batch_first=True)
		attention_mask_batch = torch.nn.utils.rnn.pad_sequence([attention_mask[torch.tensor(tokens)].to(device) for tokens, _ in valid_batch_contexts], batch_first=True)

		with torch.no_grad():
			outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)

		# Get the last hidden state
		last_hidden_state = outputs.last_hidden_state

		# Extract the embeddings of the current words (last token in each window)
		for j, (_, token_idx) in enumerate(valid_batch_contexts):
			relative_token_idx = token_idx - valid_batch_contexts[j][0][0]
			if 0 <= relative_token_idx < last_hidden_state.size(1):
				current_word_embedding = last_hidden_state[j, relative_token_idx, :].unsqueeze(0)
				all_embeddings.append(current_word_embedding)
			else:
				print(f"Skipping out-of-bounds embedding: relative_token_idx={relative_token_idx}, last_hidden_state.size(1)={last_hidden_state.size(1)}")

	# Concatenate all batches to get the final embeddings
	if all_embeddings:
		all_embeddings = torch.cat(all_embeddings, dim=0)

		# Convert the final tensor to a NumPy array
		all_embeddings_np = all_embeddings.cpu().numpy()
		print(len(all_embeddings_np))
	else:
		all_embeddings_np = np.array([])

	# Ensure the output has the same number of embeddings as the original number of words
	embedding_dim = all_embeddings_np.shape[1] if all_embeddings_np.size > 0 else model.config.hidden_size
	zero_vector = np.zeros((1, embedding_dim))

	final_embeddings_np = []
	trimmed_word_idx = 0

	for idx in range(len(words)):
		if idx in empty_word_indices:
			final_embeddings_np.append(zero_vector)
		else:
			final_embeddings_np.append(all_embeddings_np[trimmed_word_idx])
			trimmed_word_idx += 1

	final_embeddings_np = np.vstack(final_embeddings_np)

	return final_embeddings_np

def get_llama_w_prefix_same(allstories, x, k, subject, model, tokenizer):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	prefix = """Question: Repeat the following phrase: The bustling city streets, alive with the honks of impatient drivers and the hurried footsteps of commuters, contrasted sharply with the tranquil park, where children played and the elderly relaxed on benches, savoring the calm
Answer: The bustling city streets, alive with the honks of impatient drivers and the hurried footsteps of commuters, contrasted sharply with the tranquil park, where children played and the elderly relaxed on benches, savoring the calm.

Question: Repeat the following phrase: As the autumn leaves fell gracefully from the trees, their vibrant hues of red, orange, and yellow painted the ground, creating a breathtaking tapestry that celebrated the beauty and transience of nature.
Answer: As the autumn leaves fell gracefully from the trees, their vibrant hues of red, orange, and yellow painted the ground, creating a breathtaking tapestry that celebrated the beauty and transience of nature.

Question: Repeat the following phrase: Beneath the twinkling stars, the couple strolled hand in hand along the beach, the rhythmic sound of the waves crashing against the shore providing a soothing backdrop to their whispered conversations and shared laughter.
Answer: Beneath the twinkling stars, the couple strolled hand in hand along the beach, the rhythmic sound of the waves crashing against the shore providing a soothing backdrop to their whispered conversations and shared laughter.

Question: Repeat the following phrase:"""

	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding_w_prefix((wordseqs[story]).data, k, prefix, model, tokenizer)
		vectors[story] =  word_vectors
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_llama_w_prefix_and_suffix(allstories, x, k, subject, model, tokenizer):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	prefix = "This sentence: “"
	suffix = " means in one word: “"

	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding_w_prefix_suffix((wordseqs[story]).data, k, prefix, suffix, model, tokenizer)
		vectors[story] =  word_vectors
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_llama_w_prefix_att(allstories, x, k, subject, model, tokenizer):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	prefix = """Repeat the parts of the following phrase that you pay attention to most:"""

	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding_w_prefix((wordseqs[story]).data, k, prefix, model, tokenizer)
		vectors[story] =  word_vectors
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_last_token_embedding_w_prefix(words, window_size, prefix, model, tokenizer, batch_size=32):
	if torch.cuda.is_available():  # ROCm uses the same torch.cuda API
		device = torch.device('cuda')
		print(f"Using GPU: {torch.cuda.get_device_name(0)}")
		sys.stdout.flush()
	else:
		device = torch.device('cpu')
		print("ROCm not available, using CPU.")
		sys.stdout.flush()
		
	# Manually set padding token if it's not set
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	# Record indices of empty words and remove them from the list
	empty_word_indices = [idx for idx, word in enumerate(words) if not word]
	trimmed_words = [word for word in words if word]

	# Tokenize the prefix
	prefix_encoded = tokenizer(prefix, return_tensors='pt', padding=False, truncation=False)
	prefix_input_ids = prefix_encoded['input_ids'][0].to(device)
	prefix_attention_mask = prefix_encoded['attention_mask'][0].to(device)

	# Tokenize the entire text once
	encoded_input = tokenizer(trimmed_words, return_tensors='pt', padding=True, truncation=False, is_split_into_words=True)
	input_ids = encoded_input['input_ids'][0].to(device)
	attention_mask = encoded_input['attention_mask'][0].to(device)

	# Tokenize trimmed words separately to get word indices
	word_to_token_indices = []
	current_token_idx = 0
	for word in trimmed_words:
		tokenized_word = tokenizer.tokenize(word)
		token_indices = list(range(current_token_idx, current_token_idx + len(tokenized_word)))
		word_to_token_indices.append(token_indices)
		current_token_idx += len(tokenized_word)

	all_embeddings = []

	# Prepare context windows based on word indices
	context_indices = []
	for idx, token_indices in enumerate(word_to_token_indices):
		start_idx = max(0, idx - window_size + 1)
		end_idx = idx + 1
		context_window_tokens = [tok for word_idx in range(start_idx, end_idx) for tok in word_to_token_indices[word_idx]]
		context_indices.append((context_window_tokens, token_indices[-1]))

	# Process context windows in batches
	for i in tqdm(range(0, len(context_indices), batch_size)):
		batch_contexts = context_indices[i:i + batch_size]

		# Ensure no out-of-bounds tokens before creating input batches
		valid_batch_contexts = []
		for tokens, token_idx in batch_contexts:
			if all(0 <= t < len(input_ids) for t in tokens):
				valid_batch_contexts.append((tokens, token_idx))
			else:
				print(f"Skipping out-of-bounds context: tokens={tokens}, token_idx={token_idx}")
				print(len(input_ids))

		if not valid_batch_contexts:
			continue

		# Create input batches with the prefix
		input_ids_batch = []
		attention_mask_batch = []
		for tokens, _ in valid_batch_contexts:
			context_input_ids = torch.cat([prefix_input_ids, input_ids[torch.tensor(tokens).to(device)]])
			context_attention_mask = torch.cat([prefix_attention_mask, attention_mask[torch.tensor(tokens).to(device)]])
			input_ids_batch.append(context_input_ids)
			attention_mask_batch.append(context_attention_mask)

		input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True).to(device)
		attention_mask_batch = torch.nn.utils.rnn.pad_sequence(attention_mask_batch, batch_first=True).to(device)

		with torch.no_grad():
			outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)

		# Get the last hidden state
		last_hidden_state = outputs.last_hidden_state

		# Extract the embeddings of the current words (last token in each window)
		for j, (_, token_idx) in enumerate(valid_batch_contexts):
			relative_token_idx = token_idx + len(prefix_input_ids) - valid_batch_contexts[j][0][0]
			if 0 <= relative_token_idx < last_hidden_state.size(1):
				current_word_embedding = last_hidden_state[j, relative_token_idx, :].unsqueeze(0)
				all_embeddings.append(current_word_embedding)
			else:
				print(f"Skipping out-of-bounds embedding: relative_token_idx={relative_token_idx}, last_hidden_state.size(1)={last_hidden_state.size(1)}")

	# Concatenate all batches to et the final embeddings
	if all_embeddings:
		all_embeddings = torch.cat(all_embeddings, dim=0)

		# Convert the final tensor to a NumPy array
		all_embeddings_np = all_embeddings.cpu().numpy()
		print(len(all_embeddings_np))
	else:
		all_embeddings_np = np.array([])

	# Ensure the output has the same number of embeddings as the original number of words
	embedding_dim = all_embeddings_np.shape[1] if all_embeddings_np.size > 0 else model.config.hidden_size
	zero_vector = np.zeros((1, embedding_dim))

	final_embeddings_np = []
	trimmed_word_idx = 0

	for idx in range(len(words)):
		if idx in empty_word_indices:
			final_embeddings_np.append(zero_vector)
		else:
			final_embeddings_np.append(all_embeddings_np[trimmed_word_idx])
			trimmed_word_idx += 1

	final_embeddings_np = np.vstack(final_embeddings_np)

	return final_embeddings_np


def get_last_token_embedding_w_prefix_suffix(words, window_size, prefix, suffix, model, tokenizer, batch_size=32):

	os.environ['HUGGINGFACE_TOKEN'] = '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'
	token = os.getenv('HUGGINGFACE_TOKEN')

	# Load the tokenizer and model with authentication token
	tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', use_auth_token=token)
	model = AutoModel.from_pretrained('meta-llama/Meta-Llama-3-8B', use_auth_token=token)
	model.eval()

	# Check if ROCm is available and move model to GPU
	if torch.cuda.is_available():  # ROCm uses the same torch.cuda API
		device = torch.device('cuda')
		print(f"Using GPU: {torch.cuda.get_device_name(0)}")
		sys.stdout.flush()
	else:
		device = torch.device('cpu')
		print("ROCm not available, using CPU.")
		sys.stdout.flush()

	model.to(device)

	print("DONE LOADING")
		
	# Manually set padding token if it's not set
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	# Record indices of empty words and remove them from the list
	empty_word_indices = [idx for idx, word in enumerate(words) if not word]
	trimmed_words = [word for word in words if word]

	# Tokenize the prefix
	prefix_encoded = tokenizer(prefix, return_tensors='pt', padding=False, truncation=False)
	prefix_input_ids = prefix_encoded['input_ids'][0].to(device)
	prefix_attention_mask = prefix_encoded['attention_mask'][0].to(device)

	# Tokenize the suffix
	suffix_encoded = tokenizer(suffix, return_tensors='pt', padding=False, truncation=False)
	suffix_input_ids = suffix_encoded['input_ids'][0].to(device)
	suffix_attention_mask = suffix_encoded['attention_mask'][0].to(device)

	# Tokenize the entire text once
	encoded_input = tokenizer(trimmed_words, return_tensors='pt', padding=True, truncation=False, is_split_into_words=True)
	input_ids = encoded_input['input_ids'][0].to(device)
	attention_mask = encoded_input['attention_mask'][0].to(device)

	# Tokenize trimmed words separately to get word indices
	word_to_token_indices = []
	current_token_idx = 0
	for word in trimmed_words:
		tokenized_word = tokenizer.tokenize(word)
		token_indices = list(range(current_token_idx, current_token_idx + len(tokenized_word)))
		word_to_token_indices.append(token_indices)
		current_token_idx += len(tokenized_word)

	all_embeddings = []

	# Prepare context windows based on word indices
	context_indices = []
	for idx, token_indices in enumerate(word_to_token_indices):
		start_idx = max(0, idx - window_size + 1)
		end_idx = idx + 1
		context_window_tokens = [tok for word_idx in range(start_idx, end_idx) for tok in word_to_token_indices[word_idx]]
		context_indices.append((context_window_tokens, token_indices[-1]))

	# Process context windows in batches
	for i in tqdm(range(0, len(context_indices), batch_size)):
		batch_contexts = context_indices[i:i + batch_size]

		# Ensure no out-of-bounds tokens before creating input batches
		valid_batch_contexts = []
		for tokens, token_idx in batch_contexts:
			if all(0 <= t < len(input_ids) for t in tokens):
				valid_batch_contexts.append((tokens, token_idx))
			else:
				print(f"Skipping out-of-bounds context: tokens={tokens}, token_idx={token_idx}")
				print(len(input_ids))

		if not valid_batch_contexts:
			continue

		# Create input batches with the prefix
		input_ids_batch = []
		attention_mask_batch = []
		for tokens, _ in batch_contexts:
        # Concatenate prefix, main text tokens, and suffix
			context_input_ids = torch.cat([
				prefix_input_ids, 
				input_ids[torch.tensor(tokens).to(device)],
				suffix_input_ids
			])
			context_attention_mask = torch.cat([
				prefix_attention_mask,
				attention_mask[torch.tensor(tokens).to(device)],
				suffix_attention_mask
			])
			
			input_ids_batch.append(context_input_ids)
			attention_mask_batch.append(context_attention_mask)

		input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True).to(device)
		attention_mask_batch = torch.nn.utils.rnn.pad_sequence(attention_mask_batch, batch_first=True).to(device)

		with torch.no_grad():
			outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)

		# Get the last hidden state
		last_hidden_state = outputs.last_hidden_state

		# Extract the embeddings of the current words (last token in each window)
		for j, (_, token_idx) in enumerate(valid_batch_contexts):
			relative_token_idx = token_idx + len(prefix_input_ids) - valid_batch_contexts[j][0][0]
			if 0 <= relative_token_idx < last_hidden_state.size(1):
				current_word_embedding = last_hidden_state[j, relative_token_idx, :].unsqueeze(0)
				all_embeddings.append(current_word_embedding)
			else:
				print(f"Skipping out-of-bounds embedding: relative_token_idx={relative_token_idx}, last_hidden_state.size(1)={last_hidden_state.size(1)}")

	# Concatenate all batches to et the final embeddings
	if all_embeddings:
		all_embeddings = torch.cat(all_embeddings, dim=0)

		# Convert the final tensor to a NumPy array
		all_embeddings_np = all_embeddings.cpu().numpy()
		print(len(all_embeddings_np))
	else:
		all_embeddings_np = np.array([])

	# Ensure the output has the same number of embeddings as the original number of words
	embedding_dim = all_embeddings_np.shape[1] if all_embeddings_np.size > 0 else model.config.hidden_size
	zero_vector = np.zeros((1, embedding_dim))

	final_embeddings_np = []
	trimmed_word_idx = 0

	for idx in range(len(words)):
		if idx in empty_word_indices:
			final_embeddings_np.append(zero_vector)
		else:
			final_embeddings_np.append(all_embeddings_np[trimmed_word_idx])
			trimmed_word_idx += 1

	final_embeddings_np = np.vstack(final_embeddings_np)

	return final_embeddings_np


######################################
########## GloVE Features ############
######################################

def get_glove_vectors(allstories):
	"""Get glove vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')

	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		words = (wordseqs[story]).data
		embeddings = model.encode(words)
		vectors[story] = np.array(embeddings)
	return downsample_word_vectors(allstories, vectors, wordseqs)

#####################################################
########## GloVE Features with induction ############
#####################################################

def get_glove_vectors_w_induction(allstories):
	"""Get glove vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')

	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		words = (wordseqs[story]).data
		word_vectors = np.array(model.encode(words))
		vectors[story] = induction_head(wordseqs[story], word_vectors, 300)
	return downsample_word_vectors(allstories, vectors, wordseqs)

#####################################################
########## Infinigram Features with induction ############
#####################################################

def get_incont_fuzzy_infinigram_gpt(allstories, x, k, subject, batch_size=100):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	save_location = get_save_location("/blob_data/", "/home/t-smantena/internblobdl", "incont_fuzzy_gpt_topk", "", "UTS03")

	# Setup models for both infini-gram and infini-gram-w-incontext
	lm, tokenizer = setup_model(save_location, 'incontext-fuzzy', seed=1, tokenizer_checkpoint="gpt2", checkpoint="hf_openwebtext_gpt2")
	lm_b, tokenizer_b = setup_model(save_location, 'infini-gram', seed=1, tokenizer_checkpoint="gpt2", checkpoint="hf_openwebtext_gpt2")

	# Process each story in the list
	for story in allstories:
		words = (wordseqs[story]).data
		
		# Initialize an empty list to store the vectors for this story
		story_vectors = []
		
		# Process words in batches
		for i in range(0, len(words), batch_size):
			batch = words[i:i + batch_size]
			batch_vectors = generate_embeddings(batch, lm, tokenizer, lm_b, tokenizer_b, save_location, story, 'incontext-fuzzy')
			story_vectors.append(batch_vectors)
		
		# Concatenate all batch vectors for this story
		story_vectors = np.concatenate(story_vectors, axis=0)
		
		# Save the vectors to an output file
		os.makedirs(os.path.join(save_location, story), exist_ok=True)
		pickle_file_path = join(save_location, story, 'vectors.pkl')
		with open(pickle_file_path, 'wb') as pickle_file:
			pickle.dump(np.array(story_vectors), pickle_file)
			
		vectors[story] = story_vectors

		print(story, story_vectors.shape, flush=True)

	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_incont_fuzzy_infinigram_llama(allstories, x, k, subject, batch_size=100):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	login(token='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')

	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	save_location = get_save_location("/blob_data/", "/home/t-smantena/internblobdl", "incont_fuzzy_llama", "", "UTS03")

	# Setup models for both infini-gram and infini-gram-w-incontext
	lm, tokenizer = setup_model(save_location, 'incontext-fuzzy', seed=1, tokenizer_checkpoint="llama2", checkpoint="hf_openwebtext_llama")
	lm_b, tokenizer_b = setup_model(save_location, 'infini-gram', seed=1, tokenizer_checkpoint="llama2", checkpoint="hf_openwebtext_llama")

	# Process each story in the list
	for story in allstories:
		words = (wordseqs[story]).data
		
		# Initialize an empty list to store the vectors for this story
		story_vectors = []
		
		# Process words in batches
		for i in range(0, len(words), batch_size):
			batch = words[i:i + batch_size]
			batch_vectors = generate_embeddings(batch, lm, tokenizer, lm_b, tokenizer_b, save_location, story, 'incontext-fuzzy')
			story_vectors.append(batch_vectors)
		
		# Concatenate all batch vectors for this story
		story_vectors = np.concatenate(story_vectors, axis=0)
		
		# Save the vectors to an output file
		os.makedirs(os.path.join(save_location, story), exist_ok=True)
		pickle_file_path = join(save_location, story, 'vectors.pkl')
		with open(pickle_file_path, 'wb') as pickle_file:
			pickle.dump(np.array(story_vectors), pickle_file)
			
		vectors[story] = story_vectors

		print(story, story_vectors.shape, flush=True)

	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_incont_dist_infinigram_gpt(allstories, x, k, subject, batch_size=100):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	login(token='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')

	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	save_location = get_save_location("/blob_data/", "/home/t-smantena/internblobdl", "incont_dist_gpt", "", "UTS03")

	# Setup models for both infini-gram and infini-gram-w-incontext
	lm, tokenizer = setup_model(save_location, 'incontext-fuzzy', seed=1, tokenizer_checkpoint="gpt2_dist", checkpoint="hf_openwebtext_gpt2",  DATA_DIR=DATA_DIR)
	lm_b, tokenizer_b = setup_model(save_location, 'infini-gram', seed=1, tokenizer_checkpoint="gpt2", checkpoint="hf_openwebtext_gpt2", DATA_DIR=DATA_DIR)

	# Process each story in the list
	for story in allstories:
		words = (wordseqs[story]).data
		
		# Initialize an empty list to store the vectors for this story
		story_vectors = []
		
		# Process words in batches
		for i in range(0, len(words), batch_size):
			batch = words[i:i + batch_size]
			batch_vectors = generate_embeddings(batch, lm, tokenizer, lm_b, tokenizer_b, save_location, story, 'incontext-fuzzy')
			story_vectors.append(batch_vectors)
		
		# Concatenate all batch vectors for this story
		story_vectors = np.concatenate(story_vectors, axis=0)
		
		# Save the vectors to an output file
		os.makedirs(os.path.join(save_location, story), exist_ok=True)
		pickle_file_path = join(save_location, story, 'vectors.pkl')
		with open(pickle_file_path, 'wb') as pickle_file:
			pickle.dump(np.array(story_vectors), pickle_file)
			
		vectors[story] = story_vectors

		print(story, story_vectors.shape, flush=True)

	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_incont_dist_infinigram_llama(allstories, x, k, subject, batch_size=100):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	login(token='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')

	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	save_location = get_save_location("/blob_data/", "/home/t-smantena/internblobdl", "incont_dist_llama", "", "UTS03")

	# Setup models for both infini-gram and infini-gram-w-incontext
	lm, tokenizer = setup_model(save_location, 'incontext-fuzzy', seed=1, tokenizer_checkpoint="llama2_dist", checkpoint="hf_openwebtext_llama", DATA_DIR=DATA_DIR)
	lm_b, tokenizer_b = setup_model(save_location, 'infini-gram', seed=1, tokenizer_checkpoint="llama2", checkpoint="hf_openwebtext_llama", DATA_DIR=DATA_DIR)

	# Process each story in the list
	for story in allstories:
		words = (wordseqs[story]).data
		
		# Initialize an empty list to store the vectors for this story
		story_vectors = []
		
		# Process words in batches
		for i in range(0, len(words), batch_size):
			batch = words[i:i + batch_size]
			batch_vectors = generate_embeddings(batch, lm, tokenizer, lm_b, tokenizer_b, save_location, story, 'incontext-fuzzy')
			story_vectors.append(batch_vectors)
		
		# Concatenate all batch vectors for this story
		story_vectors = np.concatenate(story_vectors, axis=0)
		
		# Save the vectors to an output file
		os.makedirs(os.path.join(save_location, story), exist_ok=True)
		pickle_file_path = join(save_location, story, 'vectors.pkl')
		with open(pickle_file_path, 'wb') as pickle_file:
			pickle.dump(np.array(story_vectors), pickle_file)
			
		vectors[story] = story_vectors

		print(story, story_vectors.shape, flush=True)

	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_infinigram(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	save_location = get_save_location("/blob_data/", "/home/t-smantena/internblobdl", "infinigram", "", "UTS03")
	lm, tokenizer = setup_model(save_location, 'infini-gram', seed=1, tokenizer_checkpoint="gpt2", checkpoint="hf_openwebtext_gpt2")
	lm_b, tokenizer_b = setup_model(save_location, 'infini-gram', seed=1, tokenizer_checkpoint="gpt2", checkpoint="hf_openwebtext_gpt2")
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		words = (wordseqs[story]).data
		vectors[story] = generate_embeddings(words, lm, tokenizer, lm_b, tokenizer_b, save_location, story, 'infini')
		print(story, vectors[story].shape)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_infinigram_w_cont(allstories, x, k, subject, batch_size=100):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.
		batch_size: Number of words to process at a time.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	save_location = get_save_location("/blob_data/", "/home/t-smantena/internblobdl", "infinigram_w_cont", "", "UTS03")

	# Setup models for both infini-gram and infini-gram-w-incontext
	lm, tokenizer = setup_model(save_location, 'infini-gram-w-incontext', seed=1, tokenizer_checkpoint="gpt2", checkpoint="hf_openwebtext_gpt2")
	lm_b, tokenizer_b = setup_model(save_location, 'infini-gram', seed=1, tokenizer_checkpoint="gpt2", checkpoint="hf_openwebtext_gpt2")

	# Process each story in the list
	for story in allstories:
		words = (wordseqs[story]).data
		
		# Initialize an empty list to store the vectors for this story
		story_vectors = []
		
		# Process words in batches
		for i in range(0, len(words), batch_size):
			batch = words[i:i + batch_size]
			batch_vectors = generate_embeddings(batch, lm, tokenizer, lm_b, tokenizer_b, save_location, story, 'infini-gram-w-incontext')
			story_vectors.append(batch_vectors)
		
		# Concatenate all batch vectors for this story
		story_vectors = np.concatenate(story_vectors, axis=0)
		
		# Save the vectors to an output file
		os.makedirs(os.path.join(save_location, story), exist_ok=True)
		pickle_file_path = join(save_location, story, 'vectors2.pkl')
		with open(pickle_file_path, 'wb') as pickle_file:
			pickle.dump(np.array(story_vectors), pickle_file)
			
		vectors[story] = story_vectors

		print(story, story_vectors.shape)

	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_incont_infinigram(allstories, x, k, subject, batch_size=100):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	save_location = get_save_location("/blob_data/", "/home/t-smantena/internblobdl", "incont_infinigram", "", "UTS03")

	# Setup models for both infini-gram and infini-gram-w-incontext
	lm, tokenizer = setup_model(save_location, 'incontext-infini-gram', seed=1, tokenizer_checkpoint="gpt2", checkpoint="hf_openwebtext_gpt2")
	lm_b, tokenizer_b = setup_model(save_location, 'infini-gram', seed=1, tokenizer_checkpoint="gpt2", checkpoint="hf_openwebtext_gpt2")

	# Process each story in the list
	for story in allstories:
		words = (wordseqs[story]).data
		
		# Initialize an empty list to store the vectors for this story
		story_vectors = []
		
		# Process words in batches
		for i in range(0, len(words), batch_size):
			batch = words[i:i + batch_size]
			batch_vectors = generate_embeddings(batch, lm, tokenizer, lm_b, tokenizer_b, save_location, story, 'incontext-infini-gram')
			story_vectors.append(batch_vectors)
		
		# Concatenate all batch vectors for this story
		story_vectors = np.concatenate(story_vectors, axis=0)
		
		# Save the vectors to an output file
		os.makedirs(os.path.join(save_location, story), exist_ok=True)
		pickle_file_path = join(save_location, story, 'vectors2.pkl')
		with open(pickle_file_path, 'wb') as pickle_file:
			pickle.dump(np.array(story_vectors), pickle_file)
			
		vectors[story] = story_vectors

		print(story, story_vectors.shape)

	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_infinigram_p(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)

	vectors = {}
	vec_location = DATA_DIR[:-5] + "/infini_pca_emb" + str(x) + "/infinigram"
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		cur_vec_file = os.path.join(vec_location, story, "vectors.pkl")
		with open(cur_vec_file, 'rb') as file:
			wor_vectors = pickle.load(file)
		vectors[story] = wor_vectors
		print(story, vectors[story].shape)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_infinigram_w_cont_p(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:cd
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	vec_location = DATA_DIR[:-5] + "/infini_pca_emb" + str(x) + "/infinigram_w_cont"
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		cur_vec_file = os.path.join(vec_location, story, "vectors.pkl")
		with open(cur_vec_file, 'rb') as file:
			wor_vectors = pickle.load(file)
		vectors[story] = wor_vectors
		print(story, vectors[story].shape)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_incont_infinigram_p(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:cd
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	vec_location = DATA_DIR[:-5] + "/infini_pca_emb" + str(x) + "/incont_infinigram"
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		cur_vec_file = os.path.join(vec_location, story, "vectors.pkl")
		with open(cur_vec_file, 'rb') as file:
			wor_vectors = pickle.load(file)
		vectors[story] = wor_vectors
		print(story, vectors[story].shape)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_incont_infinigram_gpt_p(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	vec_location = DATA_DIR[:-5] + "/infini_pca_emb" + str(x) + "/incont_fuzzy_gpt"
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		cur_vec_file = os.path.join(vec_location, story, "vectors.pkl")
		with open(cur_vec_file, 'rb') as file:
			wor_vectors = pickle.load(file)
		vectors[story] = wor_vectors
		print(story, vectors[story].shape)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_incont_infinigram_llama_p(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	vec_location = DATA_DIR[:-5] + "/infini_pca_emb" + str(x) + "/incont_fuzzy_llama"
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		cur_vec_file = os.path.join(vec_location, story, "vectors.pkl")
		with open(cur_vec_file, 'rb') as file:
			wor_vectors = pickle.load(file)
		vectors[story] = wor_vectors
		print(story, vectors[story].shape)
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_pca_tr_infini_general(allstories, x, k, subject, vec_location_suffix):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.
		x: Integer value to determine the embedding path.
		k: Top percent of vectors to keep.
		vec_location_suffix: The suffix to append to the vector location for different data variations.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	top_percent = k

	vectors = {}
	resp = {}
	vec_location = DATA_DIR[:-5] + f"/infini_pca_emb{x}/{vec_location_suffix}"
	resp_location = DATA_DIR[:-5] + "/resp_pca/100/" + subject

	# for each story, create embeddings using the wordseqs
	for story in allstories:
		# Load Infinigram Vectors
		cur_vec_file = os.path.join(vec_location, story, "vectors.pkl")
		with open(cur_vec_file, 'rb') as file:
			infini_vectors = pickle.load(file)

		# Load PCA responses
		pca_file = os.path.join(resp_location, story, "vectors.pkl")
		with open(pca_file, 'rb') as file:
			resp_vectors = pickle.load(file)
			resp[story] = resp_vectors

		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		eng_vectors = sm.data
		fin_vectors = pca_tr_infini_prep(wordseqs[story], eng_vectors, infini_vectors, resp_vectors)
		vectors[story] = fin_vectors

	print("done with initial vector creation")

	# Downsample Vectors
	downsampled = downsample_word_vectors(allstories, vectors, wordseqs)
	# Adjust them in the correct way
	new_features = dict()
	for story in allstories:
		new_vector = interp_induction(wordseqs[story], eng_vectors, infini_vectors, resp[story], downsampled[story], top_percent)
		# Validate the dimension
		print("dimension of new TR", story, new_vector.shape)
		new_features[story] = new_vector

	return new_features

def get_pca_tr_infini(allstories, x, k, subject):
    """Calls the general PCA TR function with infinigram vectors."""
    return get_pca_tr_infini_general(allstories, x, k, subject, "infinigram")


def get_pca_tr_infini_w_cont(allstories, x, k, subject):
    """Calls the general PCA TR function with infinigram_w_cont vectors."""
    return get_pca_tr_infini_general(allstories, x, k, subject, "infinigram_w_cont")


def get_pca_tr_incont_infini(allstories, x, k, subject):
    """Calls the general PCA TR function with incont_infinigram vectors."""
    return get_pca_tr_infini_general(allstories, x, k, subject, "incont_infinigram")

def get_pca_tr_incont_fuzzy_gpt(allstories, x, k, subject):
    """Calls the general PCA TR function with incont_infinigram vectors."""
    return get_pca_tr_infini_general(allstories, x, k, subject, "incont_fuzzy_gpt")

def get_pca_tr_incont_fuzzy_llama(allstories, x, k, subject):
    """Calls the general PCA TR function with incont_infinigram vectors."""
    return get_pca_tr_infini_general(allstories, x, k, subject, "incont_fuzzy_llama")

def get_pca_tr_incont_dist_gpt(allstories, x, k, subject):
    """Calls the general PCA TR function with incont_infinigram vectors."""
    return get_pca_tr_infini_general(allstories, x, k, subject, "incont_dist_gpt")

def get_pca_tr_incont_dist_llama(allstories, x, k, subject):
    """Calls the general PCA TR function with incont_infinigram vectors."""
    return get_pca_tr_infini_general(allstories, x, k, subject, "incont_dist_llama")


def get_pca_tr_random(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	top_percent = k

	vectors = {}
	resp = {}
	resp_location = DATA_DIR[:-5] + "/resp_pca/100"
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		# Load PCA responses
		pca_file = os.path.join(resp_location, story, "vectors.pkl")
		with open(pca_file, 'rb') as file:
			resp_vectors = pickle.load(file)
			resp[story] = resp_vectors

		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		eng_vectors = sm.data
		vectors[story] = eng_vectors

	print("done with vector creation")
	
	# Downsample Vectors
	downsampled = downsample_word_vectors(allstories, vectors, wordseqs)
	# Adjust them in the correct way
	new_features = dict()
	for story in allstories:
		new_vector = interp_induction_random(wordseqs[story], eng_vectors, resp[story], downsampled[story], top_percent)
		# Validate the dimension
		print("dimension of new TR", story, new_vector.shape)
		new_features[story] = new_vector

	return new_features


def get_pca_tr_exact(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	top_percent = k

	vectors = {}
	resp = {}
	resp_location = DATA_DIR[:-5] + "/resp_pca/100"
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		# Load PCA responses
		pca_file = os.path.join(resp_location, story, "vectors.pkl")
		with open(pca_file, 'rb') as file:
			resp_vectors = pickle.load(file)
			resp[story] = resp_vectors

		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		eng_vectors = sm.data
		vectors[story] = eng_vectors

	print("done with vector creation")
	
	# Downsample Vectors
	downsampled = downsample_word_vectors(allstories, vectors, wordseqs)
	# Adjust them in the correct way
	new_features = dict()
	for story in allstories:
		new_vector = interp_induction_exact(wordseqs[story], eng_vectors, resp[story], downsampled[story], top_percent)
		# Validate the dimension
		print("dimension of new TR", story, new_vector.shape)
		new_features[story] = new_vector

	return new_features

def get_pca_infini(allstories, x, k, subject):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))

	vectors = {}
	vec_location = DATA_DIR[:-5] + "/infini_pca_emb" + str(x) + "/infinigram"
	resp_location = DATA_DIR[:-5] + "/resp_pca"
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		# Load Infinigram Vectors
		cur_vec_file = os.path.join(vec_location, story, "vectors.pkl")
		with open(cur_vec_file, 'rb') as file:
			infini_vectors = pickle.load(file)

		# Load PCA responses
		pca_file = os.path.join(resp_location, story, "vectors.pkl")
		with open(pca_file, 'rb') as file:
			resp_vectors = pickle.load(file)

		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		eng_vectors = sm.data
		fin_vectors = pca_infini_induction_head(wordseqs[story], eng_vectors, infini_vectors, resp_vectors)
		vectors[story] = fin_vectors
		print(story, vectors[story].shape)
	return downsample_word_vectors(allstories, vectors, wordseqs)

############################################
########## Feature Space Creation ##########
############################################

_FEATURE_CONFIG = {
	"articulation": get_articulation_vectors,
	"phonemerate": get_phonemerate_vectors,
	"wordrate": get_wordrate_vectors,
	"eng1000": get_eng1000_vectors,
	"llama_16_words": get_llama_vectors,
	"bert" : get_bert_vectors,
	"bert_32_words" : get_bert_vectors,
	"glove" : get_glove_vectors,
	"eng1000_w_induction" : get_eng1000_vectors_w_induction,
	"eng1000_vectors_w_pred_induction" : get_eng1000_vectors_w_pred_induction,
	"eng1000_w_avg_next_induction_same_word2" : get_eng1000_w_avg_next_induction,
	"eng1000_w_avg_prev_induction_same_word4" : get_eng1000_w_avg_prev_induction,
	"eng1000_w_fuzzy_distribution_1" : get_eng1000_vectors_w_fuzzy_distribution,
	"glove_w_induction" : get_glove_vectors_w_induction,
	"perplexity" : get_perplexity_features,
	"eng1000_w_fuzzy_distribution_llama" : get_eng1000_vectors_w_fuzzy_distribution_llama,
	"eng1000_w_perplexity" : get_eng1000_and_perplexity,
	"sim_8_secs" : get_eng1000_w_fuzzy_induction,
	"sim_2k_swf" : get_eng1000_w_fuzzy_induction_s,
	"sim_2k_dwf" : get_eng1000_w_fuzzy_induction_d,
	"sim_2k_sdw" : get_eng1000_w_fuzzy_induction_sd,
	"sim_2k_aw" : get_eng1000_w_fuzzy_induction_a,
	"ind_random_2" : get_eng1000_w_random_induction_k,
	"llama_dist_8_secs" : get_eng1000_vectors_w_fuzzy_distribution_llama,
	"e_perp2" : get_eng1000_and_perplexity,
	"e1000_2x" : get_eng1000_vectors_2x,
	"random_secs" : get_eng1000_w_random_induction,
	"weighted_secs" : get_eng1000_vectors_w_weighted_induction,
	"ind_secs" : get_eng1000_vectors_w_induction,
	"llama_window" : get_llama_vectors,
	"llama_2x" : get_llama_2x,
	"llama" : get_llama,
	"llama_window_avg" : get_llama_window_avg,
	"llama_k_window_avg" : get_llama_k_window_avg,
	"llama_k_window" : get_llama_k_window,
	"llama_window_avg_b" : get_llama_window_avg_b,
	"llama_k_window_avg_b" : get_llama_k_window_avg_b,
	"llama_k_window_b" : get_llama_k_window_b,
	"eng_window" : get_eng_window_vectors,
	"eng_window_avg" : get_eng_window_avg,
	"eng_k_window_avg" : get_eng_k_window_avg,
	"eng_k_window" : get_eng_k_window,
	"eng_window_avg_b" : get_eng_window_avg_b,
	"eng_k_window_avg_b" : get_eng_k_window_avg_b,
	"eng_k_window_b" : get_eng_k_window_b,
	"eng_window_b" : get_eng_window_vectors_b,
	"eng_ind_zero" : get_eng1000_w_induction_zero,
	"eng_ind_zero_b" : get_eng1000_w_induction_zero_b,
	"eng_f_ind_zero" : get_eng1000_w_both_induction_zero,
	"eng_f_ind_zero_b" : get_eng1000_w_both_induction_zero_b,
	"llama_ind_only" : get_llama_ind_only,
	"llama_non_ind_only" : get_llama_non_ind_only,
	"eng_ind_only" : get_eng1000_ind_only,
	"eng_non_ind_only" : get_eng1000_non_ind_only,
	"llama_w_prefix_same" : get_llama_w_prefix_same,
	"llama_w_prefix_att" : get_llama_w_prefix_att,
	"llama_w_prefix_means" : get_llama_w_prefix_and_suffix,
	"infinigram" : get_infinigram,
	"infinigram_w_cont" : get_infinigram_w_cont,
	"incont_infinigram" : get_incont_infinigram,
	"infinigram_p" : get_infinigram_p,
	"infinigram_w_cont_p" : get_infinigram_w_cont_p,
	"incont_infinigram_p" : get_incont_infinigram_p,
	"incont_gpt_p" : get_incont_infinigram_gpt_p,
	"incont_llama_p" : get_incont_infinigram_llama_p,
	"pca_infini_1" : get_pca_infini,
	"pca_tr_infini" : get_pca_tr_infini,
	"pca_tr_infini_w_cont" : get_pca_tr_infini_w_cont,
	"pca_tr_incont_infini" : get_pca_tr_incont_infini,
	"incont_fuzzy_gpt_topk" : get_incont_fuzzy_infinigram_gpt,
	"incont_fuzzy_llama_topk" : get_incont_fuzzy_infinigram_llama,
	"pca_tr_incont_fuzzy_gpt" : get_pca_tr_incont_fuzzy_gpt,
	"pca_tr_incont_fuzzy_llama" : get_pca_tr_incont_fuzzy_llama,
	"pca_tr_incont_dist_gpt" : get_pca_tr_incont_dist_gpt,
	"pca_tr_incont_dist_llama" : get_pca_tr_incont_dist_llama,
	"pca_tr_random" : get_pca_tr_random,
	"pca_tr_exact" : get_pca_tr_exact,
	"incont_dist_gpt" : get_incont_dist_infinigram_gpt,
	"incont_dist_llama" : get_incont_dist_infinigram_llama,
	"pca_tr_llama_ind" : get_llama_w_tr_induction,
}

def get_feature_space(feature, *args):
	return _FEATURE_CONFIG[feature](*args)