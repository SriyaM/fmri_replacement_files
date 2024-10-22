import numpy as np
import string
import itertools as itools
from ridge_utils.DataSequence import DataSequence
from tqdm import tqdm
import torch
import random
from datetime import timedelta
import sys
from scipy.spatial.distance import cosine

DEFAULT_BAD_WORDS = frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])

def pca_infini_induction_head(ds, eng_vectors, infini_vectors, resp_vectors):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors of the next words of top k similar previous words concatenated 
    to the vector.
    """
    newdata = []
    num_words = len(ds.data)

    # Append zero vectors to the response
    # 10 at the beginning and 15 at the end
    zeros_before = np.zeros((10, 200))
    zeros_after = np.zeros((5, 200))
    resp_vectors = np.concatenate((zeros_before, resp_vectors, zeros_after), axis=0)

    # Split the indices into buckets
    # Create a map of word indices to tr_times
    ind = range(0, num_words)
    ind_chunks = np.split(ind, ds.split_inds)
    ind_to_tr = {}

    for chunk_index, chunk in enumerate(ind_chunks):
        for i in chunk:
            ind_to_tr[i] = chunk_index

    for ind, w in enumerate(ds.data):
        cur_tr_ind = ind_to_tr[ind]

        if cur_tr_ind == 0 or len(ind_chunks[cur_tr_ind - 1]) == 0:
            eligible_ind = []  # No eligible indices before the first chunk or if the previous chunk is empty
        else:
            # Get the first index of the previous chunk
            final = ind_chunks[cur_tr_ind - 1][0]
            eligible_ind = [i for i in range(0, final)]
        
        if eligible_ind:
            eligible_infinigram = [infini_vectors[idx] for idx in eligible_ind]
            cur_infini = infini_vectors[ind]
            
            # Calculate cosine similarities
            similarities = [1 - cosine(cur_infini, vec) for vec in eligible_infinigram]

            # Find the index of the vector with the highest similarity
            max_similarity_index = np.argmax(similarities)

            # Find the corresponding index in the overall list
            most_similar_index = eligible_ind[max_similarity_index]

            match_tr_ind = ind_to_tr[most_similar_index]
            print("cur_tr", cur_tr_ind, "match", match_tr_ind)
            feature_1 = resp_vectors[match_tr_ind + 1] - resp_vectors[match_tr_ind]
            feature_2 = resp_vectors[match_tr_ind + 2] - resp_vectors[match_tr_ind]
            feature_3 = resp_vectors[match_tr_ind + 3] - resp_vectors[match_tr_ind]

        else:
            print("cur_tr", cur_tr_ind, "match", "none")
            feature_1 = np.zeros(200)
            feature_2 = np.zeros(200)
            feature_3 = np.zeros(200)

        
        eng1000_vec = eng_vectors[ind]
        #final_vector = np.concatenate((eng1000_vec, feature_1))
        final_vector = eng1000_vec
        newdata.append(final_vector)

    return np.array(newdata)


def interp_induction_llama(ds, eng_vectors, infini_vectors, resp_vectors, tr_vectors, top_percent = 0.2):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors of the next words of top k similar previous words concatenated 
    to the vector.
    """

    print("shape of down vec", tr_vectors.shape)
    print("shape of response vec", resp_vectors.shape)

    zeros_before = np.zeros((10, 100))
    zeros_after = np.zeros((5, 100))
    resp_vectors = np.concatenate((zeros_before, resp_vectors, zeros_after), axis=0)

    newdata = []
    num_words = len(ds.data)
    sim_length = 3218

    # Create map from index to tr_time
    ind = range(0, num_words)
    ind_chunks = np.split(ind, ds.split_inds)
    ind_to_tr = {}

    for chunk_index, chunk in enumerate(ind_chunks):
        for i in chunk:
            ind_to_tr[i] = chunk_index

    # Find the maximum value among the top dimension for each
    # Append to overall list
    # Initialize an empty list to store the max values
    max_values = []
    max_indices = []
    
    # Iterate over each row in the matrix
    for row in tr_vectors:
        # Get the last `num_values` values
        last_values = row[-sim_length:]
        # Find the maximum value in these last `num_values`
        max_value = np.max(last_values)
        # Find the index of the maximum value within the last `num_values`
        max_index = np.argmax(last_values)
        # Append the max value and its index to their respective lists
        max_values.append(max_value)
        max_indices.append(max_index)
    
    threshold = np.percentile(max_values, 100 * (1 - top_percent))
    
    # for each vector
    # If its value is in the top percent
    # Find the corresponding TR value
    # Append the relevant features
    new_array = []
    for i, row in enumerate(tr_vectors):
        current_max = max_values[i]
        # If current max is in the top x percent
        if current_max >= threshold:
            match_tr_ind = ind_to_tr[max_indices[i]]
            print("cur_tr", i, "match", match_tr_ind)
            # In case the match is near the end
            if match_tr_ind + 3 < len(resp_vectors):
                feature_1 = resp_vectors[match_tr_ind + 1] - resp_vectors[match_tr_ind]
                feature_2 = resp_vectors[match_tr_ind + 2] - resp_vectors[match_tr_ind]
                feature_3 = resp_vectors[match_tr_ind + 3] - resp_vectors[match_tr_ind]
            else:
                feature_1 = np.zeros(100)
                feature_2 = np.zeros(100)
                feature_3 = np.zeros(100)
        else:
            print("cur_tr", i, "match", "none")
            feature_1 = np.zeros(100)
            feature_2 = np.zeros(100)
            feature_3 = np.zeros(100)
        
        new_row = np.concatenate((row[:4096], feature_1, feature_2, feature_3))
        new_array.append(new_row)
    
    # Go through all the vectors and append relevant information from previous tr_times
    final_array = []
    for i, row in enumerate(new_array):
        if i > 0:
            prev_row_1 = new_array[i-1][-300:]
        else:
            prev_row_1 = np.zeros(300)

        if i > 1:
            prev_row_2 = new_array[i-2][-300:]
        else:
            prev_row_2 = np.zeros(300)

        final_row = np.concatenate((row, prev_row_1, prev_row_2))
        final_array.append(final_row)
    
    return np.array(final_array) 

def interp_induction(ds, eng_vectors, infini_vectors, resp_vectors, tr_vectors, top_percent = 0.2):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors of the next words of top k similar previous words concatenated 
    to the vector.
    """

    print("shape of down vec", tr_vectors.shape)
    print("shape of response vec", resp_vectors.shape)

    zeros_before = np.zeros((10, 100))
    zeros_after = np.zeros((5, 100))
    resp_vectors = np.concatenate((zeros_before, resp_vectors, zeros_after), axis=0)

    newdata = []
    num_words = len(ds.data)
    sim_length = 3218

    # Create map from index to tr_time
    ind = range(0, num_words)
    ind_chunks = np.split(ind, ds.split_inds)
    ind_to_tr = {}

    for chunk_index, chunk in enumerate(ind_chunks):
        for i in chunk:
            ind_to_tr[i] = chunk_index

    # Find the maximum value among the top dimension for each
    # Append to overall list
    # Initialize an empty list to store the max values
    max_values = []
    max_indices = []
    
    # Iterate over each row in the matrix
    for row in tr_vectors:
        # Get the last `num_values` values
        last_values = row[-sim_length:]
        # Find the maximum value in these last `num_values`
        max_value = np.max(last_values)
        # Find the index of the maximum value within the last `num_values`
        max_index = np.argmax(last_values)
        # Append the max value and its index to their respective lists
        max_values.append(max_value)
        max_indices.append(max_index)
    
    threshold = np.percentile(max_values, 100 * (1 - top_percent))
    
    # for each vector
    # If its value is in the top percent
    # Find the corresponding TR value
    # Append the relevant features
    new_array = []
    for i, row in enumerate(tr_vectors):
        current_max = max_values[i]
        # If current max is in the top x percent
        if current_max >= threshold:
            match_tr_ind = ind_to_tr[max_indices[i]]
            print("cur_tr", i, "match", match_tr_ind)
            # In case the match is near the end
            if match_tr_ind + 3 < len(resp_vectors):
                feature_1 = resp_vectors[match_tr_ind + 1] - resp_vectors[match_tr_ind]
                feature_2 = resp_vectors[match_tr_ind + 2] - resp_vectors[match_tr_ind]
                feature_3 = resp_vectors[match_tr_ind + 3] - resp_vectors[match_tr_ind]
            else:
                feature_1 = np.zeros(100)
                feature_2 = np.zeros(100)
                feature_3 = np.zeros(100)
        else:
            print("cur_tr", i, "match", "none")
            feature_1 = np.zeros(100)
            feature_2 = np.zeros(100)
            feature_3 = np.zeros(100)
        
        new_row = np.concatenate((row[:985], feature_1, feature_2, feature_3))
        new_array.append(new_row)
    
    # Go through all the vectors and append relevant information from previous tr_times
    final_array = []
    for i, row in enumerate(new_array):
        if i > 0:
            prev_row_1 = new_array[i-1][-300:]
        else:
            prev_row_1 = np.zeros(300)

        if i > 1:
            prev_row_2 = new_array[i-2][-300:]
        else:
            prev_row_2 = np.zeros(300)

        final_row = np.concatenate((row, prev_row_1, prev_row_2))
        final_array.append(final_row)
    
    return np.array(final_array) 

def interp_induction_random(ds, eng_vectors, resp_vectors, tr_vectors, top_percent=0.2):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors of the next words of randomly chosen previous words concatenated 
    to the vector.
    """

    print("shape of down vec", tr_vectors.shape)
    print("shape of response vec", resp_vectors.shape)

    zeros_before = np.zeros((10, 100))
    zeros_after = np.zeros((5, 100))
    resp_vectors = np.concatenate((zeros_before, resp_vectors, zeros_after), axis=0)

    num_words = len(ds.data)
    
    # Create map from index to tr_time
    ind = range(0, num_words)
    ind_chunks = np.split(ind, ds.split_inds)
    ind_to_tr = {}

    for chunk_index, chunk in enumerate(ind_chunks):
        for i in chunk:
            ind_to_tr[i] = chunk_index

    # Using random.choices to randomly select indices with replacement
    random_indices = random.choices(range(len(tr_vectors)), k=len(tr_vectors))

    new_array = []
    for i, row in enumerate(tr_vectors):
        # Choose a random index
        random_ind = random_indices[i]
        match_tr_ind = ind_to_tr.get(random_ind, None)
        
        if match_tr_ind is not None and match_tr_ind + 3 < len(resp_vectors):
            feature_1 = resp_vectors[match_tr_ind + 1] - resp_vectors[match_tr_ind]
            feature_2 = resp_vectors[match_tr_ind + 2] - resp_vectors[match_tr_ind]
            feature_3 = resp_vectors[match_tr_ind + 3] - resp_vectors[match_tr_ind]
            print("cur_tr", i, "random match", match_tr_ind)
        else:
            print("cur_tr", i, "random match", "none")
            feature_1 = np.zeros(100)
            feature_2 = np.zeros(100)
            feature_3 = np.zeros(100)

        # Concatenate the selected row and the new features
        new_row = np.concatenate((row[:985], feature_1, feature_2, feature_3))
        new_array.append(new_row)

    # Go through all the vectors and append relevant information from previous tr_times
    final_array = []
    for i, row in enumerate(new_array):
        if i > 0:
            prev_row_1 = new_array[i-1][-300:]
        else:
            prev_row_1 = np.zeros(300)

        if i > 1:
            prev_row_2 = new_array[i-2][-300:]
        else:
            prev_row_2 = np.zeros(300)

        final_row = np.concatenate((row, prev_row_1, prev_row_2))
        final_array.append(final_row)

    return np.array(final_array)


def find_matching_index(word_list, current_idx, n):
    """Finds the index of the closest match of a sequence of `n` words
    before the current index `current_idx` in the word_list.
    
    Args:
        word_list: List of words in the story.
        current_idx: Index of the word to start matching.
        n: Number of words to match.
    
    Returns:
        The index of the matching sequence if found, otherwise None.
    """
    if current_idx < n:
        return None

    # Current sequence of `n` words
    current_sequence = word_list[current_idx - n:current_idx]

    # Search for matching sequence in the earlier part of the list
    for i in range(current_idx - n):
        if word_list[i:i + n] == current_sequence:
            return i

    return None


def interp_induction_exact(ds, eng_vectors, resp_vectors, tr_vectors, top_percent=0.2):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors of the next words of matched previous word sequences
    concatenated to the vector.
    """

    print("shape of down vec", tr_vectors.shape)
    print("shape of response vec", resp_vectors.shape)

    zeros_before = np.zeros((10, 100))
    zeros_after = np.zeros((5, 100))
    resp_vectors = np.concatenate((zeros_before, resp_vectors, zeros_after), axis=0)

    num_words = len(ds.data)
    word_list = ds.data
    
    # Create map from index to tr_time
    ind = range(0, num_words)
    ind_chunks = np.split(ind, ds.split_inds)
    ind_to_tr = {}

    for chunk_index, chunk in enumerate(ind_chunks):
        for i in chunk:
            ind_to_tr[i] = chunk_index  # Mapping word index to tr index

    new_array = []
    for i, row in enumerate(tr_vectors):
        # Try to find matching sequence, start from 4 words and go down to 1
        match_word_ind = None
        for n in range(4, 0, -1):
            match_word_ind = find_matching_index(word_list, i, n)
            if match_word_ind is not None:
                break

        # If we found a match, map the word index to the corresponding tr index
        if match_word_ind is not None:
            match_tr_ind = ind_to_tr.get(match_word_ind, None)
        else:
            match_tr_ind = None
        
        # If we have a valid match_tr_ind and it's within bounds, calculate features
        if match_tr_ind is not None and match_tr_ind + 3 < len(resp_vectors):
            feature_1 = resp_vectors[match_tr_ind + 1] - resp_vectors[match_tr_ind]
            feature_2 = resp_vectors[match_tr_ind + 2] - resp_vectors[match_tr_ind]
            feature_3 = resp_vectors[match_tr_ind + 3] - resp_vectors[match_tr_ind]
            print(f"cur_tr {i} matched {n}-word sequence at index {match_word_ind} (tr index {match_tr_ind})")
        else:
            print(f"cur_tr {i} no match")
            feature_1 = np.zeros(100)
            feature_2 = np.zeros(100)
            feature_3 = np.zeros(100)

        # Concatenate the selected row and the new features
        new_row = np.concatenate((row[:985], feature_1, feature_2, feature_3))
        new_array.append(new_row)

    # Go through all the vectors and append relevant information from previous tr_times
    final_array = []
    for i, row in enumerate(new_array):
        if i > 0:
            prev_row_1 = new_array[i-1][-300:]
        else:
            prev_row_1 = np.zeros(300)

        if i > 1:
            prev_row_2 = new_array[i-2][-300:]
        else:
            prev_row_2 = np.zeros(300)

        final_row = np.concatenate((row, prev_row_1, prev_row_2))
        final_array.append(final_row)

    return np.array(final_array)


 
def pca_tr_infini_prep(ds, eng_vectors, infini_vectors, resp_vectors):
    """Creates DataSequence object with word_vectors that have the similarity scores of 
    the current word to the eligible previous words appended to the vector.
    """
    newdata = []
    num_words = len(ds.data)
    sim_vec_length = 3218

    # Split the indices into buckets
    # Create a map of word indices to tr_times
    ind = range(0, num_words)
    ind_chunks = np.split(ind, ds.split_inds)
    ind_to_tr = {}

    for chunk_index, chunk in enumerate(ind_chunks):
        for i in chunk:
            ind_to_tr[i] = chunk_index

    for ind, w in enumerate(ds.data):
        cur_tr_ind = ind_to_tr[ind]

        # Initialize similarity vector with zeros
        sim_vector = np.zeros(sim_vec_length)
        
        if cur_tr_ind == 0 or len(ind_chunks[cur_tr_ind - 3]) == 0 or cur_tr_ind < 3:
            eligible_ind = []  # No eligible indices before the first chunk or if the previous chunk is empty
        else:
            # Get the first index of the previous chunk
            final = ind_chunks[cur_tr_ind - 3][0]
            eligible_ind = [i for i in range(0, final)]
        
        if eligible_ind:
            eligible_infinigram = [infini_vectors[idx] for idx in eligible_ind]
            cur_infini = infini_vectors[ind]
            
            # Calculate cosine similarities
            for i, idx in enumerate(eligible_ind):
                similarity = 1 - cosine(cur_infini, eligible_infinigram[i])
                if idx < sim_vec_length:
                    sim_vector[idx] = similarity
        
        # Get the corresponding eng1000 vector
        eng1000_vec = eng_vectors[ind]
        
        # Concatenate the eng1000 vector with the sim_vector
        final_vector = np.concatenate((eng1000_vec, sim_vector))
        
        # Append the final vector to the newdata list
        newdata.append(final_vector)

    return np.array(newdata)


def get_next_word_distribution_llama(sentence, word_idx, tokenizer, model, max_length=512):
    """Get the distribution of the next word using LLaMA."""
    tokens = tokenizer.tokenize(sentence)
    
    # Adjust word_idx if necessary
    if word_idx >= len(tokens) - 1:
        word_idx = len(tokens) - 2
    
    # Truncate the tokens to max_length
    if len(tokens) > max_length:
        start_index = max(0, word_idx - max_length // 2)
        end_index = min(len(tokens), start_index + max_length)
        tokens = tokens[start_index:end_index]
        
        # Adjust word_idx to match the truncated sequence
        if word_idx >= max_length:
            word_idx = max_length - 2
        else:
            word_idx -= start_index
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')  # Move to GPU

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        outputs = model(tokens_tensor)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

    # Ensure word_idx + 1 is within bounds
    if word_idx + 1 >= logits.shape[1]:
        word_idx = logits.shape[1] - 2

    predicted_probabilities = torch.softmax(logits[0, word_idx + 1], dim=-1)
    return predicted_probabilities.cpu().numpy()  # Move to CPU and convert to numpy array


def find_top_k_similar_llama(prev_distributions, current_distribution, k):
    """Find the top k similar previous word distributions to the current distribution."""
    similarities = [np.dot(prev_distribution, current_distribution) for prev_distribution in prev_distributions]
    top_k_indices = np.argsort(similarities)[-k:]
    return top_k_indices

def induction_fuzzy_distribution_llama(ds, word_vectors, size, k, time_window, tokenizer, model):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors of the next words of top k similar previous words concatenated 
    to the vector.
    """
    newdata = []
    times = ds.data_times
    num_words = len(ds.data)
    
    # Generate sentences from the data sequence
    sentence = ' '.join(ds.data)
    
    prev_distributions = []
    next_word_vectors = []

    for i, w in enumerate(tqdm(ds.data)):
        current_distribution = get_next_word_distribution_llama(sentence, i, tokenizer, model)

        # Find eligible preceding instances within the time window
        eligible_indices = [j for j in range(i) if times[i] - times[j] <= time_window]

        if eligible_indices:
            eligible_distributions = [prev_distributions[idx] for idx in eligible_indices]
            top_k_indices = find_top_k_similar_llama(eligible_distributions, current_distribution, k)
            avg_next_word_vector = np.mean([next_word_vectors[eligible_indices[idx]] for idx in top_k_indices], axis=0)
        else:
            avg_next_word_vector = word_vectors[i]

        concatenated_vector = np.concatenate((word_vectors[i], avg_next_word_vector))

        if len(concatenated_vector) != size * 2:
            print("WRONG LENGTH: ", len(concatenated_vector))

        newdata.append(concatenated_vector)

        # Update previous distributions and next word vectors
        prev_distributions.append(current_distribution)
        if i < num_words - 1:
            next_word_vectors.append(word_vectors[i + 1])
        else:
            next_word_vectors.append(word_vectors[i])

    return np.array(newdata)

def induction_head(ds, word_vectors, size, time_window=10):
    """Creates a DataSequence object with word_vectors that have the word vectors
    that come right after the previous instances of the same word within the last `time_window` seconds concatenated.
    If there are no preceding instances, the current word vector is concatenated.
    """
    plain_vectors = word_vectors
    newdata = []
    times = ds.data_times

    for i, w in enumerate(ds.data):
        vector = []
        # Find eligible preceding instances within the time window
        eligible_indices = [j for j in range(i) if times[i] - times[j] <= time_window and ds.data[j] == w]
        if eligible_indices:
            # Use the next word vector of the most recent eligible instance
            recent_index = eligible_indices[-1]
            if recent_index + 1 < len(ds.data):
                next_word_vector = plain_vectors[recent_index + 1]
                #print(w, ds.data[recent_index + 1])
            else:
                next_word_vector = plain_vectors[recent_index]  # If no next word, use the current word vector itself
            vector = np.concatenate((plain_vectors[i], next_word_vector))
        else:
            vector = np.concatenate((plain_vectors[i], plain_vectors[i]))
       
        if len(vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(vector)}")
        
        newdata.append(vector)
    
    return np.array(newdata)


# def induction_head_filter(ds, word_vectors, size, time_window=10):
#     """Creates a DataSequence object with word_vectors that have the word vectors
#     that come right after the previous instances of the same word within the last `time_window` seconds concatenated.
#     If there are no preceding instances, the current word vector is concatenated.
#     """
#     plain_vectors = word_vectors
#     times = ds.data_times
#     indices = []
#     newdata = []

#     for i, w in enumerate(ds.data):
#         vector = []
#         # Find eligible preceding instances within the time window
#         eligible_indices = [j for j in range(i) if times[i] - times[j] <= time_window and ds.data[j] == w]
#         if eligible_indices:
#             recent_index = eligible_indices[-1]
#             if recent_index + 1 < len(ds.data) and w.lower().strip("{}").strip() not in DEFAULT_BAD_WORDS:
#                 next_word_vector = plain_vectors[recent_index + 1]
#                 vector = np.concatenate((plain_vectors[i], next_word_vector))

#                 if len(vector) != size * 2:
#                     print(f"WRONG LENGTH for word {i}: {len(vector)}")

#                 newdata.append(vector)
#             else:
#                 indices.append(i)
#         else:
#             indices.append(i)
    
#     return indices, np.array(newdata)

def llama_non_ind_filter(ds, word_vectors, train_stories, test_stories, rem, size, time_window=10):
    """Creates a DataSequence object with word_vectors that have the word vectors
    that come right after the previous instances of the same word within the last `time_window` seconds concatenated.
    If there are no preceding instances, the current word vector is concatenated.
    """

    chunks = []
    
    chunks = np.split(word_vectors, ds.split_inds)

    indices_to_keep = [i for i in range(len(chunks)) if i not in rem]

    # Step 2: Filter chunks, data_times, and tr_times based on specified indices
    selected_chunks = [chunks[i] for i in indices_to_keep]

    # Step 3: Update properties
    new_vectors = [item for sublist in selected_chunks for item in sublist]

    return np.array(new_vectors)

def eng_non_ind_filter(ds, word_vectors, train_stories, test_stories, rem, size, time_window=10):
    """Creates a DataSequence object with word_vectors that have the word vectors
    that come right after the previous instances of the same word within the last `time_window` seconds concatenated.
    If there are no preceding instances, the current word vector is concatenated.
    """

    chunks = []
    
    chunks = np.split(word_vectors, ds.split_inds)

    indices_to_keep = [i for i in range(len(chunks)) if i not in rem]

    # Step 2: Filter chunks, data_times, and tr_times based on specified indices
    selected_chunks = [chunks[i] for i in indices_to_keep]

    # Step 3: Update properties
    new_vectors = [item for sublist in selected_chunks for item in sublist]

    return np.array(new_vectors)
    
def induction_head_zero(ds, word_vectors, size, time_window=10):
    """Creates a DataSequence object with word_vectors that have the word vectors
    that come right after the previous instances of the same word within the last time_window seconds concatenated.
    If there are no preceding instances, the current word vector is concatenated.
    """
    plain_vectors = word_vectors
    newdata = []
    times = ds.data_times

    for i, w in enumerate(ds.data):
        vector = []
        # Find eligible preceding instances within the time window
        eligible_indices = [j for j in range(i) if times[i] - times[j] <= time_window and ds.data[j] == w]
        if eligible_indices:
            # Use the next word vector of the most recent eligible instance
            recent_index = eligible_indices[-1]
            if recent_index + 1 < len(ds.data):
                next_word_vector = plain_vectors[recent_index + 1]
                #print(w, ds.data[recent_index + 1])
            else:
                print("USING NO WORD WHEN SHOULD BE USED")
                next_word_vector =  np.zeros(size) # If no next word, use the current word vector itself
            vector = np.concatenate((plain_vectors[i], next_word_vector))
        else:
            zero_vec = np.zeros(size)
            vector = np.concatenate((plain_vectors[i], zero_vec))
       
        if len(vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(vector)}")
        
        newdata.append(vector)
    
    return np.array(newdata)

def induction_head_zero_b(ds, word_vectors, size, time_window=10):
    """Creates a DataSequence object with word_vectors that have random word vectors 
    from the last `time_window` seconds concatenated. If there are no previous instances, 
    a zero vector is concatenated.
    """
    plain_vectors = word_vectors
    newdata = []
    times = ds.data_times
    num_words = len(ds.data)

    for i, w in enumerate(ds.data):
        vector = []
        # Find eligible preceding instances within the time window
        eligible_indices_time= [j for j in range(i) if times[i] - times[j] <= time_window]
        eligible_indices = [j for j in range(i) if times[i] - times[j] <= time_window and ds.data[j] == w]
        if eligible_indices:
            # Pick a random word vector from the eligible preceding instances
            random_index = random.choice(eligible_indices_time)
            recent_index = eligible_indices[-1]
            if recent_index + 1 < len(ds.data):
                random_word_vector = plain_vectors[random_index + 1]
            else:
                print("USING NO WORD WHEN SHOULD BE USED")
                random_word_vector = np.zeros(size)  # If no next word, use a zero vector
            vector = np.concatenate((plain_vectors[i], random_word_vector))
        else:
            zero_vec = np.zeros(size)
            vector = np.concatenate((plain_vectors[i], zero_vec))
       
        if len(vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(vector)}")
        
        newdata.append(vector)
    
    return np.array(newdata)

def cosine_similarity(v1, v2):
    """Compute the cosine similarity between two vectors."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # or np.nan, depending on how you want to handle this case
    return np.dot(v1, v2) / (norm_v1 * norm_v2)

def fuzzy_induction_both_weight_zero(ds, word_vectors, size, x, k, similarity_threshold=0.6, epsilon=1e-4):
    """
    Creates DataSequence object with word_vectors that have the average of 
    the word vectors of the words that follow the top k most similar preceding words 
    within the previous x seconds concatenated to the vector. If there are no previous 
    instances, the current word is concatenated.
    """
    newdata = []
    times = ds.data_times
    num_words = len(ds.data)

    for i, w in enumerate(ds.data):
        vector = word_vectors[i]
        
        if i == 0:
            avg_vector = np.zeros(size)
        else:
            eligible_indices = [j for j in range(i) if times[i] - times[j] <= x]
            
            if not eligible_indices:
                avg_vector = np.zeros(size)
            else:
                similarities = np.array([cosine_similarity(word_vectors[j], vector) for j in eligible_indices])
                eligible_within_threshold = [j for j, sim in zip(eligible_indices, similarities) if sim >= similarity_threshold]
                
                if not eligible_within_threshold:
                    avg_vector = np.zeros(size)
                else:
                    top_k_indices = np.argsort([similarities[eligible_indices.index(j)] for j in eligible_within_threshold])[-k:]
                    top_k_eligible_indices = [eligible_within_threshold[idx] for idx in top_k_indices]

                    distances = np.array([times[i] - times[j] for j in top_k_eligible_indices])
                    inverted_distances = 1 / (distances + epsilon)
                    
                    # Cap the inverted distances to avoid extreme values
                    max_inverted_distance = 1 / epsilon
                    inverted_distances = np.minimum(inverted_distances, max_inverted_distance)
                    
                    combined_weights = similarities[top_k_indices] * inverted_distances
                    if combined_weights.sum() > 0:
                        combined_weights /= combined_weights.sum()
                    else:
                        combined_weights = np.ones_like(combined_weights) / len(combined_weights)
                    
                    next_word_vectors = []
                    for idx in top_k_eligible_indices:
                        next_index = idx + 1
                        if next_index < num_words:
                            next_word_vectors.append(word_vectors[next_index])
                    
                    if next_word_vectors:
                        next_word_vectors = np.array(next_word_vectors)
                        avg_vector = np.average(next_word_vectors, axis=0, weights=combined_weights)
                    else:
                        avg_vector = np.zeros(size)
        
        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")

        newdata.append(concatenated_vector)

    return np.array(newdata)
def fuzzy_induction_zero_b(ds, word_vectors, size, x, k, similarity_threshold=0.6, epsilon=1e-4):
    """
    Baseline function. This function creates DataSequence object with word_vectors that have the average of
    k random word vectors from all preceding words concatenated to the vector. If the original function
    would append a zero vector, this function also appends a zero vector.
    """
    newdata = []
    times = ds.data_times
    num_words = len(ds.data)

    for i, w in enumerate(ds.data):
        vector = word_vectors[i]
        
        if i == 0:
            avg_vector = np.zeros(size)
        else:
            eligible_indices = [j for j in range(i) if times[i] - times[j] <= x]
            
            # Determine if original function would append a zero vector
            if not eligible_indices:
                avg_vector = np.zeros(size)
            else:
                similarities = np.array([cosine_similarity(word_vectors[j], vector) for j in eligible_indices])
                eligible_within_threshold = [j for j, sim in zip(eligible_indices, similarities) if sim >= similarity_threshold]
                
                if not eligible_within_threshold:
                    avg_vector = np.zeros(size)
                else:
                    # Original function would have appended a word vector, so we append random vectors
                    all_indices = list(range(i))  # All preceding words
                    if len(all_indices) > k:
                        random_indices = random.sample(all_indices, k)
                    else:
                        random_indices = all_indices
                    
                    # Collect the word vectors following the randomly selected preceding words
                    next_word_vectors = []
                    for idx in random_indices:
                        next_index = idx + 1
                        if next_index < num_words:
                            next_word_vectors.append(word_vectors[next_index])
                    
                    if next_word_vectors:
                        next_word_vectors = np.array(next_word_vectors)
                        avg_vector = np.mean(next_word_vectors, axis=0)
                    else:
                        avg_vector = np.zeros(size)
        
        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")
        
        newdata.append(concatenated_vector)
    
    return np.array(newdata)

def induction_head_eng_window(ds, word_vectors, size, x, k):
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)
    
    # Create the blended vectors
    blended_vectors = []
    for i in range(num_words):
        start_idx = max(0, i - (k - 1))
        relevant_vectors = word_vectors[start_idx:i+1]
        blended_vector = np.mean(relevant_vectors, axis=0)
        blended_vectors.append(blended_vector)
    blended_vectors = np.array(blended_vectors)
    
    # Normalize blended vectors
    norm_blended_vectors = blended_vectors / (np.linalg.norm(blended_vectors, axis=1, keepdims=True) + epsilon)
    
    for i, w in enumerate(ds.data):
        original_vector = word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = np.zeros(size)
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i- x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = np.zeros(size)  # Use zero vector if no eligible indices
            else:
                # Calculate cosine similarities with the eligible preceding blended vectors
                cosine_similarities = np.dot(norm_blended_vectors[eligible_indices], norm_blended_vectors[i])
                
                # Get indices of the top 1 most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-1:]
                top_eli_ind = np.array(eligible_indices)[top_ind]
                
                # Collect the blended vectors of the top k preceding words
                valid_next_indices = top_eli_ind + k
                valid_next_indices = valid_next_indices[valid_next_indices < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = blended_vectors[valid_next_indices[0]]
                else:
                    avg_vector = np.zeros(size)  # Use zero vector if no valid indices
        
        concatenated_vector = np.concatenate((original_vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")
        
        newdata.append(concatenated_vector)
    
    return np.array(newdata)

def induction_head_eng_window_avg(ds, word_vectors, size, x, k):
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)
    
    # Create the blended vectors
    blended_vectors = []
    for i in range(num_words):
        start_idx = max(0, i - (k - 1))
        relevant_vectors = word_vectors[start_idx:i+1]
        blended_vector = np.mean(relevant_vectors, axis=0)
        blended_vectors.append(blended_vector)
    blended_vectors = np.array(blended_vectors)
    
    # Normalize blended vectors
    norm_blended_vectors = blended_vectors / (np.linalg.norm(blended_vectors, axis=1, keepdims=True) + epsilon)
    
    for i, w in enumerate(ds.data):
        original_vector = word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = np.zeros(size)
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i- x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = np.zeros(size)  # Use zero vector if no eligible indices
            else:
                # Calculate cosine similarities with the eligible preceding blended vectors
                cosine_similarities = np.dot(norm_blended_vectors[eligible_indices], norm_blended_vectors[i])
                
                # Get indices of the top 1 most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-1:]
                top_eli_ind = np.array(eligible_indices)[top_ind]
                
                # Collect the blended vectors of the top k preceding words
                valid_next_indices = top_eli_ind + k
                valid_next_indices = valid_next_indices[valid_next_indices < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = blended_vectors[valid_next_indices[0]]
                else:
                    avg_vector = np.zeros(size)  # Use zero vector if no valid indices
        
        averaged_vector = (original_vector + avg_vector) / 2
        if np.array_equal(avg_vector, np.zeros(size)):
            averaged_vector = original_vector
        if len(averaged_vector) != size:
            print(f"WRONG LENGTH for word {i}: {len(averaged_vector)}")
        
        newdata.append(averaged_vector)
    
    return np.array(newdata)



def induction_head_eng_k_window_avg(ds, word_vectors, size, x, k, num_top_similar=3):
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)
    
    # Create the blended vectors
    blended_vectors = []
    for i in range(num_words):
        start_idx = max(0, i - (k - 1))
        relevant_vectors = word_vectors[start_idx:i+1]
        blended_vector = np.mean(relevant_vectors, axis=0)
        blended_vectors.append(blended_vector)
    blended_vectors = np.array(blended_vectors)
    
    # Normalize blended vectors
    norm_blended_vectors = blended_vectors / (np.linalg.norm(blended_vectors, axis=1, keepdims=True) + epsilon)
    
    for i, w in enumerate(ds.data):
        original_vector = word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = np.zeros(size)
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i- x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = np.zeros(size)  # Use zero vector if no eligible indices
            else:
                # Calculate cosine similarities with the eligible preceding blended vectors
                cosine_similarities = np.dot(norm_blended_vectors[eligible_indices], norm_blended_vectors[i])

                # Get indices of the top 1 most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-num_top_similar:]
                top_eli_ind = np.array(eligible_indices)[top_ind]
                
                # Collect the blended vectors of the top k preceding words
                valid_next_indices = top_eli_ind + k
                valid_next_indices = valid_next_indices[valid_next_indices < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = np.mean(blended_vectors[valid_next_indices], axis=0)
                else:
                    avg_vector = np.zeros(size)  # Use zero vector if no valid indices
        
        averaged_vector = (original_vector + avg_vector) / 2
        if np.array_equal(avg_vector, np.zeros(size)):
            averaged_vector = original_vector
        if len(averaged_vector) != size:
            print(f"WRONG LENGTH for word {i}: {len(averaged_vector)}")
        
        newdata.append(averaged_vector)
    
    return np.array(newdata)


def induction_head_eng_k_window(ds, word_vectors, size, x, k, num_top_similar=3):
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)
    
    # Create the blended vectors
    blended_vectors = []
    for i in range(num_words):
        start_idx = max(0, i - (k - 1))
        relevant_vectors = word_vectors[start_idx:i+1]
        blended_vector = np.mean(relevant_vectors, axis=0)
        blended_vectors.append(blended_vector)
    blended_vectors = np.array(blended_vectors)
    
    # Normalize blended vectors
    norm_blended_vectors = blended_vectors / (np.linalg.norm(blended_vectors, axis=1, keepdims=True) + epsilon)
    
    for i, w in enumerate(ds.data):
        original_vector = word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = np.zeros(size)
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i- x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = np.zeros(size)  # Use zero vector if no eligible indices
            else:
                # Calculate cosine similarities with the eligible preceding blended vectors
                cosine_similarities = np.dot(norm_blended_vectors[eligible_indices], norm_blended_vectors[i])

                # Get indices of the top 1 most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-num_top_similar:]
                top_eli_ind = np.array(eligible_indices)[top_ind]
                
                # Collect the blended vectors of the top k preceding words
                valid_next_indices = top_eli_ind + k
                valid_next_indices = valid_next_indices[valid_next_indices < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = np.mean(blended_vectors[valid_next_indices], axis=0)
                else:
                    avg_vector = np.zeros(size)  # Use zero vector if no valid indices
        
        concatenated_vector = np.concatenate((original_vector, avg_vector))
        if len(concatenated_vector) != size:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")
        
        newdata.append(concatenated_vector)
    
    return np.array(newdata)

def induction_head_eng_k_window_baseline(ds, word_vectors, size, x, k, num_top_similar=3):
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)
    
    # Create the blended vectors
    blended_vectors = []
    for i in range(num_words):
        start_idx = max(0, i - (k - 1))
        relevant_vectors = word_vectors[start_idx:i+1]
        blended_vector = np.mean(relevant_vectors, axis=0)
        blended_vectors.append(blended_vector)
    blended_vectors = np.array(blended_vectors)
    
    # Normalize blended vectors
    norm_blended_vectors = blended_vectors / (np.linalg.norm(blended_vectors, axis=1, keepdims=True) + epsilon)
    
    zero_vector_count = 0
    
    for i, w in enumerate(ds.data):
        original_vector = word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = np.zeros(size)
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = np.zeros(size)  # Use zero vector if no eligible indices
                zero_vector_count += 1
            else:
                # Calculate cosine similarities with the eligible preceding blended vectors
                cosine_similarities = np.dot(norm_blended_vectors[eligible_indices], norm_blended_vectors[i])

                # Get indices of the top num_top_similar most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-num_top_similar:]
                top_eli_ind = np.array(eligible_indices)[top_ind]
                
                # Collect the blended vectors of the top k preceding words
                valid_next_indices = top_eli_ind + k
                valid_next_indices = valid_next_indices[valid_next_indices < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = np.mean(blended_vectors[valid_next_indices], axis=0)
                else:
                    avg_vector = np.zeros(size)  # Use zero vector if no valid indices
                    zero_vector_count += 1
        
        concatenated_vector = np.concatenate((original_vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")
        
        newdata.append(concatenated_vector)
    
    # Create baseline vectors
    baseline_data = []
    for i, w in enumerate(ds.data):
        original_vector = word_vectors[i]
        
        if i == 0 or np.array_equal(newdata[i][size:], np.zeros(size)):
            avg_vector = np.zeros(size)
        else:
            # Randomly select blended vectors
            random_indices = np.random.choice(range(num_words), num_top_similar, replace=False)
            avg_vector = np.mean(blended_vectors[random_indices], axis=0)
        
        concatenated_vector = np.concatenate((original_vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")
        
        baseline_data.append(concatenated_vector)
    
    return np.array(baseline_data)


def induction_head_eng_k_window_avg_baseline(ds, word_vectors, size, x, k, num_top_similar=3):
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)
    
    # Create the blended vectors
    blended_vectors = []
    for i in range(num_words):
        start_idx = max(0, i - (k - 1))
        relevant_vectors = word_vectors[start_idx:i+1]
        blended_vector = np.mean(relevant_vectors, axis=0)
        blended_vectors.append(blended_vector)
    blended_vectors = np.array(blended_vectors)
    
    # Normalize blended vectors
    norm_blended_vectors = blended_vectors / (np.linalg.norm(blended_vectors, axis=1, keepdims=True) + epsilon)
    
    zero_vector_count = 0
    
    for i, w in enumerate(ds.data):
        original_vector = word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = np.zeros(size)
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = np.zeros(size)  # Use zero vector if no eligible indices
                zero_vector_count += 1
            else:
                # Calculate cosine similarities with the eligible preceding blended vectors
                cosine_similarities = np.dot(norm_blended_vectors[eligible_indices], norm_blended_vectors[i])

                # Get indices of the top 1 most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-num_top_similar:]
                top_eli_ind = np.array(eligible_indices)[top_ind]
                
                # Collect the blended vectors of the top k preceding words
                valid_next_indices = top_eli_ind + k
                valid_next_indices = valid_next_indices[valid_next_indices < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = np.mean(blended_vectors[valid_next_indices], axis=0)
                else:
                    avg_vector = np.zeros(size)  # Use zero vector if no valid indices
                    zero_vector_count += 1
        
        averaged_vector = (original_vector + avg_vector) / 2
        if np.array_equal(avg_vector, np.zeros(size)):
            averaged_vector = original_vector
        if len(averaged_vector) != size:
            print(f"WRONG LENGTH for word {i}: {len(averaged_vector)}")
        
        newdata.append(averaged_vector)
    
    # Create baseline vectors
    baseline_data = []
    for i, w in enumerate(ds.data):
        original_vector = word_vectors[i]
        
        if i == 0 or np.array_equal(newdata[i], original_vector):
            avg_vector = np.zeros(size)
            zero_vector_count -= 1
        else:
            # Randomly select blended vectors
            random_indices = np.random.choice(range(num_words), num_top_similar, replace=False)
            avg_vector = np.mean(blended_vectors[random_indices], axis=0)
        
        averaged_vector = (original_vector + avg_vector) / 2
        if np.array_equal(avg_vector, np.zeros(size)):
            averaged_vector = original_vector
        if len(averaged_vector) != size:
            print(f"WRONG LENGTH for word {i}: {len(averaged_vector)}")
        
        baseline_data.append(averaged_vector)
    
    return np.array(baseline_data)


def induction_head_eng_window_baseline(ds, word_vectors, size, x, k, num_top_similar=1):
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)
    
    # Create the blended vectors
    blended_vectors = []
    for i in range(num_words):
        start_idx = max(0, i - (k - 1))
        relevant_vectors = word_vectors[start_idx:i+1]
        blended_vector = np.mean(relevant_vectors, axis=0)
        blended_vectors.append(blended_vector)
    blended_vectors = np.array(blended_vectors)
    
    # Normalize blended vectors
    norm_blended_vectors = blended_vectors / (np.linalg.norm(blended_vectors, axis=1, keepdims=True) + epsilon)
    
    zero_vector_count = 0
    
    # for i, w in enumerate(ds.data):
    #     original_vector = word_vectors[i]
        
    #     if i == 0:
    #         # If first word, concatenate with itself
    #         avg_vector = np.zeros(size)
    #     else:
    #         # Consider only preceding words within the previous x words
    #         eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
    #         if not eligible_indices:
    #             avg_vector = np.zeros(size)  # Use zero vector if no eligible indices
    #             zero_vector_count += 1
    #         else:
    #             # Calculate cosine similarities with the eligible preceding blended vectors
    #             cosine_similarities = np.dot(norm_blended_vectors[eligible_indices], norm_blended_vectors[i])

    #             # Get indices of the top num_top_similar most similar preceding words within the eligible indices
    #             top_ind = np.argsort(cosine_similarities)[-num_top_similar:]
    #             top_eli_ind = np.array(eligible_indices)[top_ind]
                
    #             # Collect the blended vectors of the top k preceding words
    #             valid_next_indices = top_eli_ind + k
    #             valid_next_indices = valid_next_indices[valid_next_indices < num_words]
    #             if valid_next_indices.size > 0:
    #                 avg_vector = np.mean(blended_vectors[valid_next_indices], axis=0)
    #             else:
    #                 avg_vector = np.zeros(size)  # Use zero vector if no valid indices
    #                 zero_vector_count += 1
        
    #     concatenated_vector = np.concatenate((original_vector, avg_vector))
    #     if len(concatenated_vector) != size * 2:
    #         print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")
        
    #     newdata.append(concatenated_vector)
    
    # Create baseline vectors
    baseline_data = []
    for i, w in enumerate(ds.data):
        original_vector = word_vectors[i]
        indices = [j for j in range(i - x, i - k + 1) if j >= 0]

        avg_vector = np.zeros(size)

        if indices:
            random_indices = np.random.choice(indices, num_top_similar, replace=False)
            avg_vector = np.mean(blended_vectors[random_indices], axis=0)
        
        concatenated_vector = np.concatenate((original_vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")
        
        baseline_data.append(concatenated_vector)
    
    return np.array(baseline_data)


def induction_head_eng_window_avg_baseline(ds, word_vectors, size, x, k, num_top_similar=1):
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)
    
    # Create the blended vectors
    blended_vectors = []
    for i in range(num_words):
        start_idx = max(0, i - (k - 1))
        relevant_vectors = word_vectors[start_idx:i+1]
        blended_vector = np.mean(relevant_vectors, axis=0)
        blended_vectors.append(blended_vector)
    blended_vectors = np.array(blended_vectors)
    
    # Normalize blended vectors
    norm_blended_vectors = blended_vectors / (np.linalg.norm(blended_vectors, axis=1, keepdims=True) + epsilon)
    
    zero_vector_count = 0
    
    for i, w in enumerate(ds.data):
        original_vector = word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = np.zeros(size)
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = np.zeros(size)  # Use zero vector if no eligible indices
                zero_vector_count += 1
            else:
                # Calculate cosine similarities with the eligible preceding blended vectors
                cosine_similarities = np.dot(norm_blended_vectors[eligible_indices], norm_blended_vectors[i])

                # Get indices of the top 1 most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-num_top_similar:]
                top_eli_ind = np.array(eligible_indices)[top_ind]
                
                # Collect the blended vectors of the top k preceding words
                valid_next_indices = top_eli_ind + k
                valid_next_indices = valid_next_indices[valid_next_indices < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = np.mean(blended_vectors[valid_next_indices], axis=0)
                else:
                    avg_vector = np.zeros(size)  # Use zero vector if no valid indices
                    zero_vector_count += 1
        
        averaged_vector = (original_vector + avg_vector) / 2
        if np.array_equal(avg_vector, np.zeros(size)):
            averaged_vector = original_vector
        if len(averaged_vector) != size:
            print(f"WRONG LENGTH for word {i}: {len(averaged_vector)}")
        
        newdata.append(averaged_vector)
    
    # Create baseline vectors
    baseline_data = []
    for i, w in enumerate(ds.data):
        original_vector = word_vectors[i]
        
        if i == 0 or np.array_equal(newdata[i], original_vector):
            avg_vector = np.zeros(size)
            zero_vector_count -= 1
        else:
            # Randomly select blended vectors
            random_indices = np.random.choice(range(num_words), num_top_similar, replace=False)
            avg_vector = np.mean(blended_vectors[random_indices], axis=0)
        
        averaged_vector = (original_vector + avg_vector) / 2
        if np.array_equal(avg_vector, np.zeros(size)):
            averaged_vector = original_vector
        if len(averaged_vector) != size:
            print(f"WRONG LENGTH for word {i}: {len(averaged_vector)}")
        
        baseline_data.append(averaged_vector)
    
    return np.array(baseline_data)
                

def induction_head_eng_filter(ds, word_vectors, size, x, k, top_percent=20):
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)
    similarities = []
    
    # Create the blended vectors
    blended_vectors = []
    for i in range(num_words):
        start_idx = max(0, i - (k - 1))
        relevant_vectors = word_vectors[start_idx:i+1]
        blended_vector = np.mean(relevant_vectors, axis=0)
        blended_vectors.append(blended_vector)
    blended_vectors = np.array(blended_vectors)
    
    # Normalize blended vectors
    norm_blended_vectors = blended_vectors / (np.linalg.norm(blended_vectors, axis=1, keepdims=True) + epsilon)
    
    for i, w in enumerate(ds.data):
        original_vector = word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = np.zeros(size)
            similarities.append(0)
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i- x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = np.zeros(size)  # Use zero vector if no eligible indices
                similarities.append(0)
            else:
                # Calculate cosine similarities with the eligible preceding blended vectors
                cosine_similarities = np.dot(norm_blended_vectors[eligible_indices], norm_blended_vectors[i])
                
                # Get indices of the top 1 most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-1:]
                top_eli_ind = np.array(eligible_indices)[top_ind]

                top_similarity = cosine_similarities[top_ind][0]
                similarities.append(abs(top_similarity))
                
                # Collect the blended vectors of the top k preceding words
                valid_next_indices = top_eli_ind + k
                valid_next_indices = valid_next_indices[valid_next_indices < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = blended_vectors[valid_next_indices[0]]
                else:
                    avg_vector = np.zeros(size)  # Use zero vector if no valid indices
        
        concatenated_vector = np.concatenate((original_vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")
        
        newdata.append(concatenated_vector)
    
    data = np.array(newdata)
    new_data, include_chunks = percent_filter(ds, similarities, data, top_percent)
    
    return np.array(new_data), include_chunks
    
def induction_head_llama(ds, word_vectors, size, x, k):
    # Find the previous instances
    # If they exist and it is in range, append the i + kth word vector.
    # Else append the same thing
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)

    wor = 0

    # Normalize word vectors
    norm_word_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + epsilon)

    for i, w in enumerate(ds.data):
        vector = norm_word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = word_vectors[i]
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = word_vectors[i]
            else:
                # Calculate cosine similarities with the eligible preceding words
                cosine_similarities = np.dot(norm_word_vectors[eligible_indices], vector)
                
                # Get indices of the top 1 most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-1:]
                top_eli_ind = np.array(eligible_indices)[top_ind]
                #print(i, top_eli_ind)

                # Collect the next word vectors following the top k preceding words
                next_index = top_eli_ind + k
                valid_next_indices = next_index[next_index < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = word_vectors[valid_next_indices[0]]
                    wor += 1
                else:
                    avg_vector = word_vectors[i]  # Fall back to the current word vector if no next words

        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")
        
        newdata.append(concatenated_vector)

    return np.array(newdata)

def induction_head_llama_filtered(ds, word_vectors, size, x, k, top_percent = 20):
    # Find the previous instances
    # If they exist and it is in range, append the i + kth word vector.
    # Else append the same thing
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)
    similarities = []

    wor = 0

    # Normalize word vectors
    norm_word_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + epsilon)

    for i, w in enumerate(ds.data):
        vector = norm_word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = word_vectors[i]
            similarities.append(0)
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = word_vectors[i]
                similarities.append(0)
            else:
                # Calculate cosine similarities with the eligible preceding words
                cosine_similarities = np.dot(norm_word_vectors[eligible_indices], vector)
                
                # Get indices of the top 1 most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-1:]
                top_eli_ind = np.array(eligible_indices)[top_ind]
                #print(i, top_eli_ind)

                top_similarity = cosine_similarities[top_ind][0]
                similarities.append(abs(top_similarity))

                # Collect the next word vectors following the top k preceding words
                next_index = top_eli_ind + k
                valid_next_indices = next_index[next_index < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = word_vectors[valid_next_indices[0]]
                    wor += 1
                else:
                    avg_vector = word_vectors[i]  # Fall back to the current word vector if no next words

        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")
        
        newdata.append(concatenated_vector)
    
    data = np.array(newdata)
    new_data, include_chunks = percent_filter(ds, similarities, data, top_percent)

    return new_data, include_chunks

def percent_filter(ds, sims, data, top_percent=20):
    new_data = []
    chunks = np.split(data, ds.split_inds)  # Split based on input data
    #print("perc_split_inds", ds.split_inds[0:20])
    sim_chunks = np.split(sims, ds.split_inds)
    include_chunks_ind = []

    # Calculate the average similarity for each chunk
    avg_sims = []
    for sim_chunk in sim_chunks:
        if len(sim_chunk) > 0:
            avg_sim = sum(sim_chunk) / len(sim_chunk)
        else:
            avg_sim = 0
        avg_sims.append(avg_sim)
    
    # Determine the threshold for the top x percent
    threshold_index = int(len(avg_sims) * (top_percent / 100.0))
    if threshold_index <= 0:
        threshold_index = 1
    
    sorted_avg_sims = sorted(avg_sims, reverse=True)
    thresh = sorted_avg_sims[threshold_index - 1] if sorted_avg_sims else 0

    # Filter based on the top x percent threshold
    cur = 0
    for i, chunk in enumerate(chunks):
        avg_sim = avg_sims[i]
        if avg_sim >= thresh:
            include_chunks_ind.append(i)
            new_data.extend(chunk)  # Collect the entire chunk
        cur += len(chunk)
    
    #print(f"Included Chunks Len_filt", len(include_chunks_ind))
    #print(f"Filtered Data Length_filt: {len(new_data)}")

    return np.array(new_data), include_chunks_ind

def induction_head_llama_eng(ds, word_vectors, eng_vecs, size, x, k):
    # Find the previous instances
    # If they exist and it is in range, append the i + kth word vector.
    # Else append the same thing
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)

    wor = 0

    # Normalize word vectors
    norm_word_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + epsilon)

    for i, w in enumerate(ds.data):
        vector = eng_vecs[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = eng_vecs[i]
            dot_vector = norm_word_vectors[i]
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = eng_vecs[i]
            else:
                # Calculate cosine similarities with the eligible preceding words
                cosine_similarities = np.dot(norm_word_vectors[eligible_indices], dot_vector)
                
                # Get indices of the top 1 most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-1:]
                top_eli_ind = np.array(eligible_indices)[top_ind]
                #print(i, top_eli_ind)

                # Collect the next word vectors following the top k preceding words
                next_index = top_eli_ind + k
                valid_next_indices = next_index[next_index < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = eng_vecs[valid_next_indices[0]]
                    wor += 1
                else:
                    avg_vector = eng_vecs[i]  # Fall back to the current word vector if no next words

        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")
        
        newdata.append(concatenated_vector)

    return np.array(newdata)

def induction_head_llama_avg(ds, word_vectors, size, x, k):
    # Find the previous instances
    # If they exist and it is in range, append the i + kth word vector.
    # Else append the same thing
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)

    wor = 0

    # Normalize word vectors
    norm_word_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + epsilon)

    for i, w in enumerate(ds.data):
        vector = norm_word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = word_vectors[i]
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = word_vectors[i]
            else:
                # Calculate cosine similarities with the eligible preceding words
                cosine_similarities = np.dot(norm_word_vectors[eligible_indices], vector)
                
                # Get indices of the top 1 most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-1:]
                top_eli_ind = np.array(eligible_indices)[top_ind]
                #print(i, top_eli_ind)

                # Collect the next word vectors following the top k preceding words
                next_index = top_eli_ind + k
                valid_next_indices = next_index[next_index < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = word_vectors[valid_next_indices[0]]
                    wor += 1
                else:
                    avg_vector = word_vectors[i]  # Fall back to the current word vector if no next words

        averaged_vector = (vector + avg_vector) / 2
        if len(averaged_vector) != size:
            print(f"WRONG LENGTH for word {i}: {len(averaged_vector)}")
        
        newdata.append(averaged_vector)

    return np.array(newdata)

def induction_head_llama_k_avg(ds, word_vectors, size, x, k, num_top_similar=3):
    # Find the previous instances
    # If they exist and it is in range, append the i + kth word vector.
    # Else append the same thing
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)

    wor = 0

    # Normalize word vectors
    norm_word_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + epsilon)

    for i, w in enumerate(ds.data):
        vector = norm_word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = word_vectors[i]
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = word_vectors[i]
            else:
                # Calculate cosine similarities with the eligible preceding words
                cosine_similarities = np.dot(norm_word_vectors[eligible_indices], vector)
                
                # Get indices of the top num_top_similar most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-num_top_similar:]
                top_eli_ind = np.array(eligible_indices)[top_ind]
                #print(i, top_eli_ind)

                # Collect the next word vectors following the top k preceding words
                next_indices = top_eli_ind + k
                valid_next_indices = next_indices[next_indices < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = np.mean(word_vectors[valid_next_indices], axis=0)
                    wor += 1
                else:
                    avg_vector = word_vectors[i]  # Fall back to the current word vector if no next words

        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")
        
        newdata.append(concatenated_vector)

    return np.array(newdata)

def induction_head_llama_k(ds, word_vectors, size, x, k, num_top_similar=3):
    # Find the previous instances
    # If they exist and it is in range, append the i + kth word vector.
    # Else append the same thing
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)

    wor = 0

    # Normalize word vectors
    norm_word_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + epsilon)

    for i, w in enumerate(ds.data):
        vector = norm_word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = word_vectors[i]
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = word_vectors[i]
            else:
                # Calculate cosine similarities with the eligible preceding words
                cosine_similarities = np.dot(norm_word_vectors[eligible_indices], vector)
                
                # Get indices of the top num_top_similar most similar preceding words within the eligible indices
                top_ind = np.argsort(cosine_similarities)[-num_top_similar:]
                top_eli_ind = np.array(eligible_indices)[top_ind]
                #print(i, top_eli_ind)

                # Collect the next word vectors following the top k preceding words
                next_indices = top_eli_ind + k
                valid_next_indices = next_indices[next_indices < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = np.mean(word_vectors[valid_next_indices], axis=0)
                    wor += 1
                else:
                    avg_vector = word_vectors[i]  # Fall back to the current word vector if no next words

        averaged_vector = (vector + avg_vector) / 2
        if len(averaged_vector) != size:
            print(f"WRONG LENGTH for word {i}: {len(averaged_vector)}")
        
        newdata.append(averaged_vector)

    return np.array(newdata)


def induction_head_llama_avg_baseline(ds, word_vectors, size, x, k):
    # Find the previous instances
    # If they exist and it is in range, append the i + kth word vector.
    # Else append the same thing
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)

    # Normalize word vectors
    norm_word_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + epsilon)

    for i, w in enumerate(ds.data):
        vector = norm_word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = word_vectors[i]
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = word_vectors[i]
            else:
                # Choose a random preceding word within the eligible indices
                random_ind = random.choice(eligible_indices)
                
                # Collect the next word vector following the random preceding word
                next_index = random_ind + k
                if next_index < num_words:
                    avg_vector = word_vectors[next_index]
                else:
                    avg_vector = word_vectors[i]  # Fall back to the current word vector if no next words

        averaged_vector = (vector + avg_vector) / 2
        if len(averaged_vector) != size:
            print(f"WRONG LENGTH for word {i}: {len(averaged_vector)}")
        
        newdata.append(averaged_vector)

    return np.array(newdata)

def induction_head_llama_k_avg_baseline(ds, word_vectors, size, x, k, num_top_similar=3):
    # Find the previous instances
    # If they exist and it is in range, append the i + kth word vector.
    # Else append the same thing
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)

    # Normalize word vectors
    norm_word_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + epsilon)

    for i, w in enumerate(ds.data):
        vector = norm_word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = word_vectors[i]
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = word_vectors[i]
            else:
                # Choose random preceding words within the eligible indices
                random_inds = random.sample(eligible_indices, min(num_top_similar, len(eligible_indices)))
                
                # Collect the next word vectors following the random preceding words
                next_indices = np.array(random_inds) + k
                valid_next_indices = next_indices[next_indices < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = np.mean(word_vectors[valid_next_indices], axis=0)
                else:
                    avg_vector = word_vectors[i]  # Fall back to the current word vector if no next words

        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")
        
        newdata.append(concatenated_vector)

    return np.array(newdata)

def induction_head_llama_k_baseline(ds, word_vectors, size, x, k, num_top_similar=3):
    # Find the previous instances
    # If they exist and it is in range, append the i + kth word vector.
    # Else append the same thing
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)

    # Normalize word vectors
    norm_word_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + epsilon)

    for i, w in enumerate(ds.data):
        vector = norm_word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = word_vectors[i]
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = word_vectors[i]
            else:
                # Choose random preceding words within the eligible indices
                random_inds = random.sample(eligible_indices, min(num_top_similar, len(eligible_indices)))
                
                # Collect the next word vectors following the random preceding words
                next_indices = np.array(random_inds) + k
                valid_next_indices = next_indices[next_indices < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = np.mean(word_vectors[valid_next_indices], axis=0)
                else:
                    avg_vector = word_vectors[i]  # Fall back to the current word vector if no next words

        averaged_vector = (vector + avg_vector) / 2
        if len(averaged_vector) != size:
            print(f"WRONG LENGTH for word {i}: {len(averaged_vector)}")
        
        newdata.append(averaged_vector)

    return np.array(newdata)

def induction_head_llama_k_baseline(ds, word_vectors, size, x, k, num_top_similar=3):
    # Find the previous instances
    # If they exist and it is in range, append the i + kth word vector.
    # Else append the same thing
    epsilon = 1e-6
    newdata = []
    num_words = len(ds.data)

    # Normalize word vectors
    norm_word_vectors = word_vectors / (np.linalg.norm(word_vectors, axis=1, keepdims=True) + epsilon)

    for i, w in enumerate(ds.data):
        vector = norm_word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = word_vectors[i]
        else:
            # Consider only preceding words within the previous x words
            eligible_indices = [j for j in range(i - x, i - k + 1) if j >= 0]
            
            if not eligible_indices:
                avg_vector = word_vectors[i]
            else:
                # Choose random preceding words within the eligible indices
                random_inds = random.sample(eligible_indices, min(num_top_similar, len(eligible_indices)))
                
                # Collect the next word vectors following the random preceding words
                next_indices = np.array(random_inds) + k
                valid_next_indices = next_indices[next_indices < num_words]
                if valid_next_indices.size > 0:
                    avg_vector = np.mean(word_vectors[valid_next_indices], axis=0)
                else:
                    avg_vector = word_vectors[i]  # Fall back to the current word vector if no next words

        averaged_vector = (vector + avg_vector) / 2
        if len(averaged_vector) != size:
            print(f"WRONG LENGTH for word {i}: {len(averaged_vector)}")
        
        newdata.append(averaged_vector)

    return np.array(newdata)


def get_next_word_distribution(sentence, word_idx, tokenizer, model, max_length=512):
    """Get the distribution of the next word using BERT."""
    tokens = tokenizer.tokenize(sentence)
    
    # Adjust word_idx if necessary
    if word_idx >= len(tokens) - 1:
        word_idx = len(tokens) - 2
    
    tokens[word_idx + 1] = '[MASK]'
    
    # Truncate the tokens to max_length
    if len(tokens) > max_length:
        start_index = max(0, word_idx - max_length // 2)
        end_index = min(len(tokens), start_index + max_length)
        tokens = tokens[start_index:end_index]
        
        # Adjust word_idx to match the truncated sequence
        if word_idx >= max_length:
            word_idx = max_length - 2
        else:
            word_idx -= start_index
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
    segments_ids = [0] * len(tokens)
    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')  # Move to GPU
    segments_tensors = torch.tensor([segments_ids]).to('cuda')  # Move to GPU

    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs.logits

    # Ensure word_idx + 1 is within bounds
    if word_idx + 1 >= predictions.shape[1]:
        word_idx = predictions.shape[1] - 2

    predicted_probabilities = torch.softmax(predictions[0, word_idx + 1], dim=-1)
    return predicted_probabilities.cpu().numpy()  # Move to CPU and convert to numpy array


def find_top_k_similar(prev_distributions, current_distribution, k):
    """Find the top k similar previous word distributions to the current distribution."""
    similarities = [np.dot(prev_distribution, current_distribution) for prev_distribution in prev_distributions]
    top_k_indices = np.argsort(similarities)[-k:]
    return top_k_indices

def induction_fuzzy_distribution(ds: DataSequence, word_vectors, size, k, tokenizer, model):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors of the next words of top k similar previous words concatenated 
    to the vector.
    """
    newdata = []
    times = ds.data_times
    num_words = len(ds.data)
    
    # Generate sentences from the data sequence
    sentences = [' '.join(ds.data)]
    
    prev_distributions = []
    next_word_vectors = []

    for i, w in enumerate(ds.data):
        sentence = sentences[0]
        current_distribution = get_next_word_distribution(sentence, i, tokenizer, model)

        if prev_distributions:
            top_k_indices = find_top_k_similar(prev_distributions, current_distribution, k)
            avg_next_word_vector = np.mean([next_word_vectors[idx] for idx in top_k_indices], axis=0)
        else:
            avg_next_word_vector = word_vectors[i]

        concatenated_vector = np.concatenate((word_vectors[i], avg_next_word_vector))

        if len(concatenated_vector) != size * 2:
            print("WRONG LENGTH: ", len(concatenated_vector))

        newdata.append(concatenated_vector)

        # Update previous distributions and next word vectors
        prev_distributions.append(current_distribution)
        if i < num_words - 1:
            next_word_vectors.append(word_vectors[i + 1])
        else:
            next_word_vectors.append(word_vectors[i])

    return np.array(newdata)

def induction_head_random(ds: DataSequence, word_vectors, size, secs):
    """Creates DataSequence object with word_vectors that have a random previous 
    instance of preceding words within the last 10 seconds concatenated to the vector.
    If there are no preceding instances within the first 10 words, the current word is concatenated.
    """
    newdata = []
    times = ds.data_times

    print(ds.data)

    for i, w in enumerate(ds.data):
        
        if i == 0:
            # First word vector, concatenate itself
            vector = np.concatenate((word_vectors[i], word_vectors[i]))
        else:
            
            # Find all preceding vectors within the last x seconds
            eligible_indices = [j for j in range(i) if times[i] - times[j] <= secs]
            if eligible_indices:
                random_index = random.choice(eligible_indices)
                vector = np.concatenate((word_vectors[i], word_vectors[random_index]))
                print(w, ds.data[random_index], i - random_index)
            else:
                vector = np.concatenate((word_vectors[i], word_vectors[i]))

        if len(vector) != size * 2:
            print("WRONG LENGTH: ", len(vector))

        newdata.append(vector)

    return np.array(newdata)


def induction_head_random_k(ds, word_vectors, size, x):
    """
    Creates a DataSequence object with word_vectors that have an average of 
    x randomly chosen previous word vectors concatenated to the current word vector.
    If there are no preceding instances (first word), the current word is concatenated.
    """
    newdata = []
    times = ds.data_times

    for i, w in enumerate(ds.data):
        if i == 0:
            # First word vector, concatenate itself
            vector = np.concatenate((word_vectors[i], word_vectors[i]))
        else:
            # Find all preceding vectors
            eligible_indices = list(range(i))
            if eligible_indices:
                # Calculate distances with the eligible preceding words
                distances = np.array([times[i] - times[j] for j in eligible_indices])
                # Invert distances to use as weights (more recent words have higher weights)
                inverted_distances = 1 / (distances + 1e-6)  # Add small constant to avoid division by zero

                # Normalize weights
                weights = inverted_distances / inverted_distances.sum()

                # Randomly sample x preceding word vectors based on the weights
                chosen_indices = np.random.choice(eligible_indices, size=min(x, i), replace=False, p=weights)

                #print([ds.data[j] for j in chosen_indices])

                # Calculate the weighted average of chosen word vectors
                weighted_avg_vector = np.average([word_vectors[j] for j in chosen_indices], axis=0, weights=[weights[eligible_indices.index(j)] for j in chosen_indices])
                vector = np.concatenate((word_vectors[i], weighted_avg_vector))
            else:
                vector = np.concatenate((word_vectors[i], word_vectors[i]))

        if len(vector) != size * 2:
            print("WRONG LENGTH: ", len(vector))

        newdata.append(vector)

    return np.array(newdata)

def induction_head_avg_next(ds: DataSequence, word_vectors, size, x):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors over the next x seconds concatenated to the vector. If there 
    are not x seconds left, averages all remaining word vectors.
    """
    newdata = []
    times = ds.data_times
    num_words = len(ds.data)

    for i, w in enumerate(ds.data):
        vector = word_vectors[i]

        # Determine the end time for averaging
        end_time = times[i] + x
        # Find all subsequent vectors within the next x seconds
        eligible_indices = [j for j in range(i+1, num_words) if times[j] <= end_time]
        
        if eligible_indices:
            avg_vector = np.mean([word_vectors[j] for j in eligible_indices], axis=0)
        else:
            # If no remaining words, concatenate the current vector itself
            avg_vector = word_vectors[i]

        concatenated_vector = np.concatenate((vector, avg_vector))

        if len(concatenated_vector) != size * 2:
            print("WRONG LENGTH: ", len(concatenated_vector))

        newdata.append(concatenated_vector)

    return np.array(newdata)

import numpy as np

def induction_head_avg_next_same_word(ds: DataSequence, word_vectors, size, x=2):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors over the next x seconds after the last occurrence of the 
    same word concatenated to the vector. If there are no previous instances, 
    the current word is concatenated.
    """
    ind_dict = {}
    newdata = []
    times = ds.data_times
    num_words = len(ds.data)

    for i, w in enumerate(ds.data):
        vector = word_vectors[i]

        if w in ind_dict:
            last_occurrence_index = ind_dict[w]
            # Determine the end time for averaging
            end_time = times[last_occurrence_index] + x
            # Find all subsequent vectors within the next x seconds after the last occurrence
            eligible_indices = [j for j in range(last_occurrence_index + 1, num_words) if times[j] <= end_time]

            if eligible_indices:
                avg_vector = np.mean([word_vectors[j] for j in eligible_indices], axis=0)
            else:
                avg_vector = word_vectors[i]
        else:
            # If no previous occurrence, concatenate the current vector itself
            avg_vector = word_vectors[i]

        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print("WRONG LENGTH: ", len(concatenated_vector))

        newdata.append(concatenated_vector)

        # Update the last occurrence index of the word
        ind_dict[w] = i

    return np.array(newdata)

def induction_head_avg_prev(ds: DataSequence, word_vectors, size, x):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors over the previous x seconds concatenated to the vector. If there 
    are not x seconds available, averages all previous word vectors.
    """
    newdata = []
    times = ds.data_times
    num_words = len(ds.data)

    for i, w in enumerate(ds.data):
        vector = word_vectors[i]

        # Determine the start time for averaging
        start_time = times[i] - x
        # Find all preceding vectors within the previous x seconds
        eligible_indices = [j for j in range(i) if times[j] >= start_time]
        
        if eligible_indices:
            avg_vector = np.mean([word_vectors[j] for j in eligible_indices], axis=0)
        else:
            avg_vector = word_vectors[i]

        concatenated_vector = np.concatenate((vector, avg_vector))

        if len(concatenated_vector) != size * 2:
            print("WRONG LENGTH: ", len(concatenated_vector))

        newdata.append(concatenated_vector)

    return np.array(newdata)

def induction_head_avg_prev_same_word(ds: DataSequence, word_vectors, size, x):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors over the previous x seconds before the last occurrence of the 
    same word concatenated to the vector. If there are no previous instances, 
    the current word is concatenated.
    """
    ind_dict = {}
    newdata = []
    times = ds.data_times
    num_words = len(ds.data)

    for i, w in enumerate(ds.data):
        vector = word_vectors[i]

        if w in ind_dict:
            last_occurrence_index = ind_dict[w]
            # Determine the start time for averaging
            start_time = times[last_occurrence_index] - x
            # Find all preceding vectors within the previous x seconds before the last occurrence
            eligible_indices = [j for j in range(last_occurrence_index) if times[j] >= start_time]

            if eligible_indices:
                avg_vector = np.mean([word_vectors[j] for j in eligible_indices], axis=0)
            else:
                # If no eligible indices, use the vector of the current word
                avg_vector = word_vectors[i]
        else:
            # If no previous occurrence, concatenate the current vector itself
            avg_vector = word_vectors[i]

        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print("WRONG LENGTH: ", len(concatenated_vector))

        newdata.append(concatenated_vector)

        # Update the last occurrence index of the word
        ind_dict[w] = i

    return np.array(newdata)

# NEW FUNCTIONS

def fuzzy_induction_avg_weight(ds: DataSequence, word_vectors, size, x, k, epsilon=1e-6):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors of the words that follow the top k most similar preceding words 
    within the previous x seconds concatenated to the vector. If there are no previous 
    instances, the current word is concatenated.
    """
    newdata = []
    times = ds.data_times
    num_words = len(ds.data)

    for i, w in enumerate(ds.data):
        vector = word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = vector
        else:
            # Consider only preceding words within the previous x seconds
            eligible_indices = [j for j in range(i) if times[i] - times[j] <= x]
            
            if not eligible_indices:
                avg_vector = vector
            else:
                # Calculate dot products with the eligible preceding words
                dot_products = np.dot(word_vectors[eligible_indices], vector)
                # Add small constant to dot products to avoid zero weights
                adjusted_dot_products = dot_products + epsilon
                
                # Get indices of the top k most similar preceding words within the eligible indices
                top_k_indices = np.argsort(adjusted_dot_products)[-k:]
                top_k_eligible_indices = [eligible_indices[idx] for idx in top_k_indices]

                # Collect the next word vectors following the top k preceding words
                next_word_vectors = []
                for idx in top_k_eligible_indices:
                    next_index = idx + 1
                    if next_index < num_words:
                        next_word_vectors.append(word_vectors[next_index])
                
                if next_word_vectors:
                    avg_vector = np.mean(next_word_vectors, axis=0)
                else:
                    avg_vector = vector  # Fall back to the current word vector if no next words

        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")

        newdata.append(concatenated_vector)

    return np.array(newdata)

def fuzzy_induction_sim_weight(ds, word_vectors, size, x, k, epsilon=1e-6):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors of the words that follow the top k most similar preceding words 
    within the previous x seconds concatenated to the vector. If there are no previous 
    instances, the current word is concatenated.
    """
    newdata = []
    times = ds.data_times
    num_words = len(ds.data)

    for i, w in enumerate(ds.data):
        vector = word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = vector
        else:
            # Consider only preceding words within the previous x seconds
            eligible_indices = [j for j in range(i) if times[i] - times[j] <= x]
            
            if not eligible_indices:
                avg_vector = vector
            else:
                # Calculate dot products with the eligible preceding words
                dot_products = np.dot(word_vectors[eligible_indices], vector)
                # Add small constant to dot products to avoid zero weights
                adjusted_dot_products = dot_products + epsilon
                
                # Get indices of the top k most similar preceding words within the eligible indices
                top_k_indices = np.argsort(adjusted_dot_products)[-k:]
                top_k_eligible_indices = [eligible_indices[idx] for idx in top_k_indices]

                # Collect the next word vectors following the top k preceding words
                next_word_vectors = []
                next_word_weights = []
                for idx in top_k_eligible_indices:
                    next_index = idx + 1
                    if next_index < num_words:
                        next_word_vectors.append(word_vectors[next_index])
                        next_word_weights.append(adjusted_dot_products[idx])

                if next_word_vectors:
                    next_word_vectors = np.array(next_word_vectors)
                    next_word_weights = np.array(next_word_weights)
                    # Normalize weights to sum to 1
                    next_word_weights /= next_word_weights.sum()
                    # Compute the weighted average
                    avg_vector = np.average(next_word_vectors, axis=0, weights=next_word_weights)
                else:
                    avg_vector = vector  # Fall back to the current word vector if no next words

        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")

        newdata.append(concatenated_vector)

    return np.array(newdata)


def fuzzy_induction_dist_weight(ds: DataSequence, word_vectors, size, x, k, epsilon=1e-6):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors of the words that follow the top k most similar preceding words 
    within the previous x seconds concatenated to the vector. If there are no previous 
    instances, the current word is concatenated.
    """
    newdata = []
    times = ds.data_times
    num_words = len(ds.data)

    for i, w in enumerate(ds.data):
        vector = word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = vector
        else:
            # Consider only preceding words within the previous x seconds
            eligible_indices = [j for j in range(i) if times[i] - times[j] <= x]
            
            if not eligible_indices:
                avg_vector = vector
            else:
                # Calculate dot products with the eligible preceding words
                dot_products = np.dot(word_vectors[eligible_indices], vector)
                # Add small constant to dot products to avoid zero weights
                adjusted_dot_products = dot_products + epsilon
                
                # Get indices of the top k most similar preceding words within the eligible indices
                top_k_indices = np.argsort(adjusted_dot_products)[-k:]
                top_k_eligible_indices = [eligible_indices[idx] for idx in top_k_indices]

                # Calculate distances with the eligible preceding words
                distances = np.array([times[i] - times[j] for j in top_k_eligible_indices])
                # Invert distances to use as weights (more recent words have higher weights)
                inverted_distances = 1 / (distances + epsilon)  # Add small constant to avoid division by zero
                
                # Collect the next word vectors following the top k preceding words
                next_word_vectors = []
                for idx in top_k_eligible_indices:
                    next_index = idx + 1
                    if next_index < num_words:
                        next_word_vectors.append(word_vectors[next_index])
                
                if next_word_vectors:
                    next_word_vectors = np.array(next_word_vectors)
                    next_word_weights = inverted_distances / inverted_distances.sum()
                    avg_vector = np.average(next_word_vectors, axis=0, weights=next_word_weights)
                else:
                    avg_vector = vector  # Fall back to the current word vector if no next words

        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")

        newdata.append(concatenated_vector)

    return np.array(newdata)

def fuzzy_induction_both_weight(ds: DataSequence, word_vectors, size, x, k, epsilon=1e-6):
    """Creates DataSequence object with word_vectors that have the average of 
    the word vectors of the words that follow the top k most similar preceding words 
    within the previous x seconds concatenated to the vector. If there are no previous 
    instances, the current word is concatenated.
    """
    newdata = []
    times = ds.data_times
    num_words = len(ds.data)

    for i, w in enumerate(ds.data):
        vector = word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = vector
        else:
            # Consider only preceding words within the previous x seconds
            eligible_indices = [j for j in range(i) if times[i] - times[j] <= x]
            
            if not eligible_indices:
                avg_vector = vector
            else:
                # Calculate dot products with the eligible preceding words
                dot_products = np.dot(word_vectors[eligible_indices], vector)
                # Add small constant to dot products to avoid zero weights
                adjusted_dot_products = dot_products + epsilon
                
                # Get indices of the top k most similar preceding words within the eligible indices
                top_k_indices = np.argsort(adjusted_dot_products)[-k:]
                top_k_eligible_indices = [eligible_indices[idx] for idx in top_k_indices]

                # Calculate distances with the eligible preceding words
                distances = np.array([times[i] - times[j] for j in top_k_eligible_indices])
                # Invert distances to use as weights (more recent words have higher weights)
                inverted_distances = 1 / (distances + epsilon)  # Add small constant to avoid division by zero
                
                # Combine similarity and distance weights
                combined_weights = (adjusted_dot_products[top_k_indices] + epsilon) * inverted_distances
                combined_weights /= combined_weights.sum()
                
                # Collect the next word vectors following the top k preceding words
                next_word_vectors = []
                for idx in top_k_eligible_indices:
                    next_index = idx + 1
                    if next_index < num_words:
                        next_word_vectors.append(word_vectors[next_index])
                
                if next_word_vectors:
                    next_word_vectors = np.array(next_word_vectors)
                    avg_vector = np.average(next_word_vectors, axis=0, weights=combined_weights)
                else:
                    avg_vector = vector  # Fall back to the current word vector if no next words
        
        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print(f"WRONG LENGTH for word {i}: {len(concatenated_vector)}")

        newdata.append(concatenated_vector)

    return np.array(newdata)


def fuzzy_induction_head(ds: DataSequence, word_vectors, size, k):
    """Creates DataSequence object with word_vectors that have the vector 
    of the word right after the top k most similar preceding words concatenated 
    to the current word vector. If there are no previous instances, the current 
    word is concatenated.
    """
    newdata = []
    num_words = len(ds.data)

    for i, w in enumerate(ds.data):
        vector = word_vectors[i]
        
        if i == 0:
            # If first word, concatenate with itself
            avg_vector = vector
        else:
            # Calculate dot products with preceding words
            dot_products = np.dot(word_vectors[:i], vector)
            # Get indices of the top k most similar preceding words
            top_k_indices = np.argsort(dot_products)[-k:]

            # Collect the vectors right after the top k preceding words
            following_vectors = []
            for idx in top_k_indices:
                if idx + 1 < num_words:
                    following_vectors.append(word_vectors[idx + 1])
                else:
                    following_vectors.append(word_vectors[idx])

            # Average the collected vectors
            avg_vector = np.mean(following_vectors, axis=0)

        concatenated_vector = np.concatenate((vector, avg_vector))
        if len(concatenated_vector) != size * 2:
            print("WRONG LENGTH: ", len(concatenated_vector))

        newdata.append(concatenated_vector)

    return np.array(newdata)


def get_next_word_seq(sequence, tokenizer, model, max_length):

    def get_last_n_words(sequence, n):
        words = sequence.split()
        return ' '.join(words[-n:])

    def predict_next_word(sequence):
        inputs = tokenizer(sequence, return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')

        with torch.no_grad():
            outputs = model(**inputs)

        next_token_logits = outputs.logits[:, -1, :]

        predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()

        # Decode the token to get the full word
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)

        # Ensure the token is a full word
        predicted_word = tokenizer.convert_tokens_to_string([predicted_token]).strip()

        return predicted_word

    # Ensure the sequence is not longer than max_length words
    sequence = get_last_n_words(sequence, max_length)

    # Check if sequence is empty
    if len(sequence) < 1:
        return ""

    predicted_words = []
    for _ in range(5):
        next_word = predict_next_word(sequence)
        predicted_words.append(next_word)
        sequence += ' ' + next_word
        sequence = get_last_n_words(sequence, max_length)

    return predicted_words

def induction_head_pred(ds: DataSequence, word_vectors, size, lsasm):
    """Creates DataSequence object with word_vectors that have previous instances of 
    preceding words concatenated to the vector. If there are no preceeding instances,
    the predicted next word of GPT-2 is used.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    model.eval()

    ind_dict = {}


    plain_vectors = word_vectors
    newdata = []

    for i, w in enumerate(tqdm(ds.data)):
        vector = []
        if w in ind_dict:
            vector = np.concatenate((plain_vectors[i], ind_dict[w]))
        else:
            # In this case attach the GPT-2 predicted next work instead of the current word
            vec_so_far = ds.data[:i+1]
            so_far = " ".join(vec_so_far)
            next_words = get_next_word_seq(so_far, tokenizer, model, 100)

            next_word = ""
            if len(next_words) > 0:
                # if first is punctuation, assign second
                # is second is also punctation, assign 3rd
                next_word = next_words[0]
                for word in next_words:
                    if word not in string.punctuation:
                        next_word = word
                        break
                
            v = []
            try:
                v = np.concatenate((v, lsasm[str.encode(next_word.lower())]))
            except KeyError as e:
                v = np.concatenate((v,plain_vectors[i])) #lsasm.data.shape[0],))

            vector = np.concatenate((plain_vectors[i], v))
        
        if i < len(ds.data) - 1:
            ind_dict[w] = plain_vectors[i+1]
        
        if len(vector) != size * 2:
            print("WRONG LENGTH: ", len(word_vectors[i]))
        
        newdata.append(vector)
    
    return np.array(newdata)

def make_word_ds(grids, trfiles, bad_words=DEFAULT_BAD_WORDS):
    """Creates DataSequence objects containing the words from each grid, with any words appearing
    in the [bad_words] set removed.
    """
    ds = dict()
    stories = list(set(trfiles.keys()) & set(grids.keys()))
    for st in stories:
        grtranscript = grids[st].tiers[1].make_simple_transcript()
        ## Filter out bad words
        goodtranscript = [x for x in grtranscript
                          if x[2].lower().strip("{}").strip() not in bad_words]
        d = DataSequence.from_grid(goodtranscript, trfiles[st][0])
        ds[st] = d

    return ds

def make_word_ds_no_filter(grids, trfiles, bad_words=DEFAULT_BAD_WORDS):
    """Creates DataSequence objects containing the words from each grid, with any words appearing
    in the [bad_words] set removed.
    """
    ds = dict()
    stories = list(set(trfiles.keys()) & set(grids.keys()))
    for st in stories:
        grtranscript = grids[st].tiers[1].make_simple_transcript()
        ## Filter out bad words
        goodtranscript = [x for x in grtranscript]
        d = DataSequence.from_grid(goodtranscript, trfiles[st][0])
        ds[st] = d

    return ds

def make_word_ds_filter(grids, trfiles, include, bad_words=DEFAULT_BAD_WORDS):
    ds = dict()
    stories = list(set(trfiles.keys()) & set(grids.keys()))
    for st in stories:
        grtranscript = grids[st].tiers[1].make_simple_transcript()
        
        inc = include[st]
        ## Filter out bad words
        goodtranscript = [x for x in grtranscript
                          if x[2].lower().strip("{}").strip() not in bad_words]

        d = DataSequence.from_grid(goodtranscript, trfiles[st][0])
        d = update_ds_filter(d, inc)
        #print("inc_len", st, len(inc))
        ds[st] = d

    return ds

def make_word_ds_non_ind_filter(grids, trfiles, test_stories, train_stories, rem, bad_words=DEFAULT_BAD_WORDS):
    ds = dict()
    stories = list(set(trfiles.keys()) & set(grids.keys()))
    for st in stories:
        grtranscript = grids[st].tiers[1].make_simple_transcript()
        
        
        ## Filter out bad words
        goodtranscript = [x for x in grtranscript
                          if x[2].lower().strip("{}").strip() not in bad_words]

        d = DataSequence.from_grid(goodtranscript, trfiles[st][0])
        if st in test_stories:
            remove = rem[st]
            all_indices = set(range(len(d.tr_times)))
            remove_indices_set = set(remove)
            include = list(all_indices - remove_indices_set)
            d = update_ds_filter(d, include)
        ds[st] = d

    return ds

def update_ds_filter(d, indices):
    chunks = np.split(d.data, d.split_inds)
    data_time_chunks = np.split(d.data_times, d.split_inds)

    # Filter chunks, data_times, and tr_times based on specified indices
    selected_chunks = [chunks[i] for i in indices]
    selected_data_time_chunks = [data_time_chunks[i] for i in indices]
    selected_tr_times = [d.tr_times[i] for i in indices]

    #print("chunk_len", len(selected_chunks))
    #print("perc_split_inds", d.split_inds[0:20])

    # for i in range(5):
    #     print(selected_chunks[i])

    # Update properties using numpy concatenate to handle arrays
    if selected_chunks:
        d.data = np.concatenate(selected_chunks)  # Flatten the list of selected chunks
        d.data_times = np.concatenate(selected_data_time_chunks)  # Flatten the list of selected data time chunks
    else:
        d.data = np.array([])
        d.data_times = np.array([])

    # Calculate the new split indices based on the length of selected chunks
    new_split_inds = [len(chunk) for chunk in selected_chunks]
    d.split_inds = np.cumsum(new_split_inds).tolist()  # Cumulative sum to get split points
    d.tr_times = selected_tr_times
    
    return d

def make_phoneme_ds(grids, trfiles):
    """Creates DataSequence objects containing the phonemes from each grid.
    """
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[0].make_simple_transcript()
        d = DataSequence.from_grid(grtranscript, trfiles[st][0])
        ds[st] = d

    return ds

phonemes = ['AA', 'AE','AH','AO','AW','AY','B','CH','D', 'DH', 'EH', 'ER', 'EY', 
            'F', 'G', 'HH', 'IH', 'IY', 'JH','K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 
            'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']

def make_character_ds(grids, trfiles):
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[2].make_simple_transcript()
        fixed_grtranscript = [(s,e,map(int, c.split(","))) for s,e,c in grtranscript if c]
        d = DataSequence.from_grid(fixed_grtranscript, trfiles[st][0])
        ds[st] = d
    return ds

def make_dialogue_ds(grids, trfiles):
    ds = dict()
    for st, gr in grids.iteritems():
        grtranscript = gr.tiers[3].make_simple_transcript()
        fixed_grtranscript = [(s,e,c) for s,e,c in grtranscript if c]
        ds[st] = DataSequence.from_grid(fixed_grtranscript, trfiles[st][0])
    return ds

def histogram_phonemes(ds, phonemeset=phonemes):
    """Histograms the phonemes in the DataSequence [ds].
    """
    olddata = ds.data
    N = len(ds.data)
    newdata = np.zeros((N, len(phonemeset)))
    phind = dict(enumerate(phonemeset))
    for ii,ph in enumerate(olddata):
        try:
            #ind = phonemeset.index(ph.upper().strip("0123456789"))
            ind = phind[ph.upper().strip("0123456789")]
            newdata[ii][ind] = 1
        except Exception as e:
            pass

    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)

def histogram_phonemes2(ds, phonemeset=phonemes):
    """Histograms the phonemes in the DataSequence [ds].
    """
    olddata = np.array([ph.upper().strip("0123456789") for ph in ds.data])
    newdata = np.vstack([olddata==ph for ph in phonemeset]).T
    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)

def make_semantic_model(ds: DataSequence, lsasms: list, sizes: list):
    """
    ds
        datasequence to operate on
    lsasms
        semantic models to use
    sizes
        sizes of resulting vectors from each semantic model
    """
    newdata = []
    num_lsasms = len(lsasms)
    for w in ds.data:
        v = []
        for i in range(num_lsasms):
            lsasm = lsasms[i]
            size = sizes[i]
            try:
                v = np.concatenate((v, lsasm[str.encode(w.lower())]))
            except KeyError as e:
                v = np.concatenate((v, np.zeros((size)))) #lsasm.data.shape[0],))
        newdata.append(v)
    return DataSequence(np.array(newdata), ds.split_inds, ds.data_times, ds.tr_times)

def make_character_model(dss):
    """Make character indicator model for a dict of datasequences.
    """
    stories = dss.keys()
    storychars = dict([(st,np.unique(np.hstack(ds.data))) for st,ds in dss.iteritems()])
    total_chars = sum(map(len, storychars.values()))
    char_inds = dict()
    ncharsdone = 0
    for st in stories:
        char_inds[st] = dict(zip(storychars[st], range(ncharsdone, ncharsdone+len(storychars[st]))))
        ncharsdone += len(storychars[st])

    charmodels = dict()
    for st,ds in dss.iteritems():
        charmat = np.zeros((len(ds.data), total_chars))
        for ti,charlist in enumerate(ds.data):
            for char in charlist:
                charmat[ti, char_inds[st][char]] = 1
        charmodels[st] = DataSequence(charmat, ds.split_inds, ds.data_times, ds.tr_times)

    return charmodels, char_inds

def make_dialogue_model(ds):
    return DataSequence(np.ones((len(ds.data),1)), ds.split_inds, ds.data_times, ds.tr_times)

def modulate(ds, vec):
    """Multiplies each row (each word/phoneme) by the corresponding value in [vec].
    """
    return DataSequence((ds.data.T*vec).T, ds.split_inds, ds.data_times, ds.tr_times)

def catmats(*seqs):
    keys = seqs[0].keys()
    return dict([(k, DataSequence(np.hstack([s[k].data for s in seqs]), seqs[0][k].split_inds)) for k in keys])
