"""
Project Part 2 - Task 1
Sarah Lawrence
Apr 3,2024
Behrooz Mansouri
470 Text Mining and Analytics
"""
import csv
import os
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
import tensorflow_hub as hub
import string
from PyPDF2 import PdfReader
import multiprocessing

# pip install PyPDF2
# pip install gensim
# pip install tensorflow tensorflow-hub

# Load the Universal Sentence Encoder (if dosn't work use: embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-lite/2"))
print("Loading embedding modle...", end='')
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
print("Done")

def tokenize(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    # Remove stop words
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def csv_reader(filepath, key_row, data_row,check):
    information_dict = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        # Collecting the csv informtion
        for row in csv_reader:
            if row:  
                key = row[key_row]  
                data = row[data_row]
                if check == True:
                    rel = row["rel"]
                else:
                    rel = "1"
            if (rel == "1"):
                if key in information_dict:
                    information_dict[key].append(data)
                else:
                    information_dict[key] = [data]
    return information_dict

def read_trec_folder(folder_path):
    all_documents = []
    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        all_documents.append(filename)
    return all_documents

def read_trec_file(file_path, documents, desc):
    docno_text_dict = {}
    file_paths = [os.path.join(file_path, doc) for doc in documents]

    for file_path in tqdm(file_paths, desc=desc):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Use regex to find DOCNO and TEXT
            docnos = re.findall(r'<DOCNO>(.*?)</DOCNO>', content, re.DOTALL)
            texts = re.findall(r'<TEXT>(.*?)</TEXT>', content, re.DOTALL)

            min_length = min(len(docnos), len(texts))
            for i in range(min_length):
                # Trim any leading/trailing whitespace
                docno = docnos[i].strip()
                text = texts[i].strip()
                # Add to the dictionary
                docno_text_dict[docno] = text
    return docno_text_dict

def compares(key_dict, text_dict):
    updated_text_dict = []
    for text_id, text_value in tqdm(text_dict.items(), desc="Training data"):
        # Iterate through key_dict values
        for key, value_list in key_dict.items():
            if text_id in value_list:
                # Storing values
                new_dict = [key, text_id, text_value,]
                updated_text_dict.append(new_dict)
                break  # Stop searching for this text_key
    return updated_text_dict

def write_dict_to_csv(updated_text_dict, filename):
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["key", "text_id", "text"])
        # Write each row with key and values
        for item in tqdm(updated_text_dict, desc="Training data"):
            writer.writerow(item)
        print("Done creating csv!")

def read_pdf(file_path):
    BDI_list = []
    pdf_reader = PdfReader(file_path)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    for BDI in re.findall(r"[3]\s+([^\.]+)\.?", pdf_text):
        modified_BDI = BDI.replace("f eel", "feel").replace("a ll", "all")
        BDI_list.append(modified_BDI)
    return BDI_list

def create_sentence_embeddings(sentences):
    tokenized_sentences = [tokenize(sentence) for sentence in sentences]
    embeddings = embed([" ".join(tokens) for tokens in tokenized_sentences])
    return embeddings.numpy()

def sentence_to_vector(word_representations):
    # Sum the word representations
    summed_representation = np.sum(word_representations, axis=0)
    # Calculate the length of the sentence
    sentence_length = len(word_representations)
    # Normalize by dividing by the square root of sentence length
    normalized_vector = summed_representation / np.sqrt(sentence_length)
    return normalized_vector

def calculate_similarity(embedding1, embedding2):
    # dot or inner
    similarity = np.dot(embedding1, embedding2)
    return similarity

def process_sentence(key_sentence_pair):
    key, sentence = key_sentence_pair
    embedding = create_sentence_embeddings([sentence])
    normalized_embedding = sentence_to_vector(embedding)
    return key, normalized_embedding

def compare_dicts(dict1, dict2):
    comparison_results = {}
    
    for key in dict1:
        # Initialize list to store matching values
        matching_values = []
        
        # Compare values of this key from dict1 with dict2
        for value in dict1[key]:
            if value in dict2.get(key, []):
                matching_values.append(value)
        
        # Add matching values to comparison_results
        comparison_results[key] = matching_values
    return comparison_results

def compare_lengths(dict1, dict2):
    length_differences = {}
    dict1_length = {}
    dict2_length = {}
    
    for key in dict1:
        # Get the lengths of values for this key in dict1 and dict2
        len_dict1 = len(dict1[key])
        len_dict2 = len(dict2.get(key, []))
        
        # Calculate the difference in lengths
        difference = len_dict1 - len_dict2
        
        # Add difference to length_differences dictionary
        length_differences[key] = difference
        dict1_length[key] = len_dict1
        dict2_length[key] = len_dict2
    
    return length_differences, dict1_length, dict2_length

def print_matching_values(dict1, dict2, output_file):
    result = compare_dicts(dict1, dict2)
    length_result,key_length,result_length = compare_lengths(dict1, result)

    with open(output_file, mode='w', newline='') as csvfile:
        fieldnames = ['Symptom_num', 'Relevant Sentences (RS)', 'Found RS','Haven\'t Found RS']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for key in length_result:
                writer.writerow({
                    'Symptom_num': key,
                    'Found RS': result_length[key],
                    'Haven\'t Found RS': length_result[key],
                    'Relevant Sentences (RS)': key_length[key]
                })

def main():
    find_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(find_directory, "new_data")

    # Task 1: Getting data
    # Getting txt file names (Compare csv ids with file ids)
    all_documents = read_trec_folder(file_path)
    # Storing the text file ids and there sentences
    all_files = read_trec_file(file_path, all_documents, "Reading All Files")
    
    # Task 2: Querie creation
    # BDI collected queries
    print("Collecting BDI information...", end='')
    file_path = os.path.join(
        find_directory, "Beck-Depression-Inventory-BDI.pdf")
    BDI_list = read_pdf(file_path)
    print("Done")

    # Task 3: Embeding modle
    # Embeding BDI 
    BDI_embeddings = []
    for bsentance in tqdm(BDI_list, desc="Embedding BDI tokens"):
        sentence_embedding = create_sentence_embeddings([bsentance])
        normalized_embedding = sentence_to_vector(sentence_embedding)
        BDI_embeddings.append(normalized_embedding)
    # TODO If you dont want to use all available CPU cores
    # sentence_embeddings = {}
    # for key, sentence in tqdm(all_files.items(), desc="Embedding sentence tokens"):
    #     embedding = create_sentence_embeddings([sentence], embed)
    #     normalized_embedding = sentence_to_vector(embedding)
    #     sentence_embeddings[key] = normalized_embedding
    
    # Embeding sentances    - Process sentence embeddings in parallel
    print("Embedding sentances...", end='')
    num_processes = multiprocessing.cpu_count() - 1  # Use all available cores except one
    pool = multiprocessing.Pool(num_processes)
    sentence_embedding_pairs = list(all_files.items())  
    sentence_embeddings = dict(pool.imap(process_sentence, sentence_embedding_pairs))
    pool.close()
    pool.join()
    print("Done")

    # Task 4: Cosine similarity
    results = []
    for i, bdi_vector in tqdm(enumerate(BDI_embeddings), desc="Cosine similarity:"):
        similarities = {}
        for sentence_id, sentence_vector in sentence_embeddings.items():
            similarities[sentence_id] = calculate_similarity(bdi_vector, sentence_vector)
        # Sort similarities based on similarity score (the second value in each tuple)
        ranked_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        # Store the top 1,000 ranked results for each BDI vector
        results.extend([(i + 1, "Q0", sentence_id, j + 1, similarity, "system_name")
                        for j, (sentence_id, similarity) in enumerate(ranked_results[:1000])])

    # Task 5: Format results to CSV
    print("Writing to CSV...", end='')
    output_filename = "results.csv"
    with open(output_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["symptom_number", "Q0", "sentence-id", "position_in_ranking", "score", "system_name"])
        for result in results:
            writer.writerow(result)
    print("Results written to", output_filename)
    
    # Task 6 Testing
    # Getting csv information 
    file_path = os.path.join(find_directory, "g_rels_consenso.csv")
    key_dict = csv_reader(file_path, "query", "docid",True)
    file_path = os.path.join(find_directory, "results.csv")
    result_info = csv_reader(file_path, "symptom_number", "sentence-id",False)
    
    print_matching_values(key_dict, result_info, "comparson.csv")
    print("Testing written to comparson.csv")
    
    print("All Done! \u001b[32m:)\u001b[32m")
   

if __name__ == "__main__":
    main()
