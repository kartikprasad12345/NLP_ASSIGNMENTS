import json
from collections import defaultdict
import re
class WordPieceTokenizer:
    def __init__(self,corpus_file_path,path_for_vocab_txt_out,iterations,max_vocab_size):
        self.corpus_file_path=corpus_file_path
        self.iterations=iterations
        self.vocabulary=None
        self.path_for_vocab_txt_out=path_for_vocab_txt_out
        self.max_vocab_size=max_vocab_size
        with open(corpus_file_path, 'r') as file:
            self.corpus=file.read()
        self.corpus=self.corpus
    def get_vocabulary(self):
        return self.vocabulary
    def extract_words(self):
        words = re.findall(r'\b[a-z]+\b', self.corpus)
        words_frequency=defaultdict(int)
        for word in words:
            words_frequency[word]+=1
        words_split={}
        for word in words_frequency.keys():
            word_split=[]
            word_split.append(word[0])
            for mid_alpha in word[1:]:
                word_split.append(f"##{mid_alpha}")
            words_split[word]=word_split
        
        print(words_split)
        print(words_frequency)
        return words_frequency,words_split
    def compute_pair_score(self, words_frequency, words_splits):
        pairs_frequency,letter_frequency= defaultdict(int), defaultdict(int)

        for word,word_frequency in words_frequency.items():
            split_of_the_word=words_splits[word]
            for i in range(len(split_of_the_word)):
                letter_frequency[split_of_the_word[i]]+=word_frequency
                if(i+1<len(split_of_the_word)):
                    pairs_frequency[(split_of_the_word[i],split_of_the_word[i+1])]+=word_frequency
        
        scores_return={}
        for pair_word, pair_frequency in pairs_frequency.items():
            scores_return[pair_word]=pair_frequency/(letter_frequency[pair_word[0]]*letter_frequency[pair_word[1]])
        return scores_return
    def merge_pair_of_words(self, letter1,letter2,words_splits):
        for word,word_split in words_splits.items():
            if len(word_split)<=1:
                continue
            else:
                word_split_modified=word_split
                ind_to_iterate=0
                while ind_to_iterate+1<len(word_split_modified):
                    if word_split_modified[ind_to_iterate]==letter1 and word_split_modified[ind_to_iterate+1]==letter2:
                        if letter2.startswith("##"):
                            merged_letters=letter1+letter2[2:]
                            word_split_modified=word_split_modified[:ind_to_iterate]+[merged_letters]+word_split_modified[ind_to_iterate+2:]
                        else:
                            merged_letters=letter1+letter2
                            word_split_modified=word_split_modified[:ind_to_iterate]+[merged_letters]+word_split_modified[ind_to_iterate+2:]
                    else:
                        ind_to_iterate+=1
                words_splits[word]=word_split_modified
        return words_splits
    

    def construct_vocabulary(self):
        words_frequency,words_split=self.extract_words()
        vocabulary_return=[]
        for words,word_split in words_split.items():
            for letters in word_split:
                if letters not in vocabulary_return:
                    vocabulary_return.append(letters)
        

        for epoch in range(self.iterations):
            scores_of_pairs=self.compute_pair_score(words_frequency,words_split)
            best_pair_temp,max_score_temp=None,None
            for pair_loop,score_loop in scores_of_pairs.items():
                if best_pair_temp==None or max_score_temp==None or max_score_temp<score_loop:
                    best_pair_temp=pair_loop
                    max_score_temp=score_loop
            if best_pair_temp!=None and max_score_temp!=None:
                words_split=self.merge_pair_of_words(best_pair_temp[0],best_pair_temp[1],words_split)
                if best_pair_temp[1].startswith("##"):
                    token_to_append=best_pair_temp[0]+best_pair_temp[1][2:]
                    if token_to_append not in vocabulary_return:
                        vocabulary_return.append(token_to_append)
                else:
                    token_to_append=best_pair_temp[0]+best_pair_temp[1]
                    if token_to_append not in vocabulary_return:
                        vocabulary_return.append(token_to_append)
            else:
                break

            if self.max_vocab_size<len(vocabulary_return):
                break
        
        vocabulary_return=vocabulary_return[:self.max_vocab_size-2]
        vocabulary_return.sort()
        vocabulary_return=["[UNK]","[PAD]"]+vocabulary_return
        self.vocabulary=vocabulary_return
        with open(self.path_for_vocab_txt_out,"w") as file:
            for tokens in self.vocabulary:
                file.write(tokens+"\n")
        
        return self.vocabulary
    

    def tokenize_helper(self,sentence):
        words_in_the_sentence=re.findall(r'\b[a-z]+\b', sentence)
        token_ret=[]


        for word in words_in_the_sentence:
                
                tokens_word=[]


                start_ind_of_word=0
                while(start_ind_of_word<len(word)):

                    found_bool=False
                    for end_ind_of_word in range(len(word),start_ind_of_word,-1):


                        if start_ind_of_word==0:
                            subword_token_to_find=word[start_ind_of_word:end_ind_of_word]

                        

                            if subword_token_to_find in self.vocabulary:
                                token_ret.append(subword_token_to_find)
                                start_ind_of_word=end_ind_of_word
                                found_bool=True
                                break
                        else:
                            subword_token_to_find="##"+word[start_ind_of_word:end_ind_of_word]

                            if subword_token_to_find in self.vocabulary:
                                token_ret.append(subword_token_to_find)
                                start_ind_of_word=end_ind_of_word
                                found_bool=True
                                break

                    if found_bool==False:
                        tokens_word=[self.vocabulary[0]]
                        break
                    
                    token_ret=token_ret+tokens_word
        return token_ret

    def tokenize(self,sentences_in_json,token_out_json):
        with open(sentences_in_json,"r") as f:
            data_in=json.load(f)
        data_out={}

        for entries in data_in:
            id_of_sen=entries["id"]
            sentence_of_id=entries["sentence"]
            data_out[id_of_sen]=self.tokenize_helper(sentence_of_id)
        
        with open(token_out_json,"w") as f:
            json.dump(data_out,f,indent=4)
        
        print(data_out)
        return data_out

        

                    

































# import json
# from collections import defaultdict
# import re
# class WordPieceTokenizer:
#     def __init__(self,corpus_file_path,path_for_vocab_txt_out,iterations,max_vocab_size):


#         """
#         Initializes the WordPiece tokenizer.

#         Parameters:
#         - corpus_file_path (str): Path to the text corpus file.
#         - path_for_vocab_txt_out (str): Path where the vocabulary output file will be stored.
#         - iterations (int): Number of iterations for the training process.
#         - max_vocab_size (int): Maximum allowed vocabulary size.
#         """

#         self.corpus_file_path=corpus_file_path
#         self.iterations=iterations
#         self.vocabulary=None
#         self.path_for_vocab_txt_out=path_for_vocab_txt_out
#         self.max_vocab_size=max_vocab_size


#         # Read the corpus file into memory
#         with open(corpus_file_path, 'r') as file:
#             self.corpus=file.read()
#         self.corpus=self.corpus
#     def get_vocabulary(self):
#         # Get the list of Vocabulary
#         return self.vocabulary
#     def extract_words(self):
#         """
#         Extracts words from the corpus, calculates word frequencies, 
#         and splits words into subword units using the WordPiece format.

#         Process:
#         1. Finds all words in the corpus using regex.
#         2. Computes the frequency of each word.
#         3. Splits each word into subword tokens using the WordPiece format, for eg-> house becomes [h,##o,##u,##s,##e0]
#         4. Prints the resulting subword splits and word frequencies.

#         Returns:
#         - None (Prints words_split and words_frequency dictionaries required for debgging) 
#         """

#         words = re.findall(r'\b[a-z]+\b', self.corpus)
#         words_frequency=defaultdict(int)
#         for word in words:
#             words_frequency[word]+=1
#         words_split={}
#         for word in words_frequency.keys():
#             word_split=[]
#             word_split.append(word[0])
#             for mid_alpha in word[1:]:
#                 word_split.append(f"##{mid_alpha}")
#             words_split[word]=word_split
        
#         print(words_split)
#         print(words_frequency)
#         return words_frequency,words_split
#     def compute_pair_score(self, words_frequency, words_splits):

#         """
#         Computes the score for adjacent subword pairs in words to determine merge priority.

#         Process:
#         1. Initializes frequency dictionaries for individual letters and letter pairs.
#         2. Iterates through the words and their frequencies:
#         - Counts occurrences of each letter in the word.
#         - Counts occurrences of adjacent letter pairs.
#         3. Calculates the score for each letter pair using:
#         - pair_frequency / (frequency of first letter * frequency of second letter)
#         4. Returns a dictionary with the computed scores.

#         Parameters:
#         - words_frequency (dict): Dictionary containing words and their corresponding frequency.
#         - words_splits (dict): Dictionary mapping words to their subword tokenization.

#         Returns:
#         - scores_return (dict): Dictionary containing scores for each adjacent letter pair.
#         """
#         pairs_frequency,letter_frequency= defaultdict(int), defaultdict(int)

#         for word,word_frequency in words_frequency.items():
#             split_of_the_word=words_splits[word]
#             for i in range(len(split_of_the_word)):
#                 letter_frequency[split_of_the_word[i]]+=word_frequency
#                 if(i+1<len(split_of_the_word)):
#                     pairs_frequency[(split_of_the_word[i],split_of_the_word[i+1])]+=word_frequency
        
#         scores_return={}
#         for pair_word, pair_frequency in pairs_frequency.items():
#             scores_return[pair_word]=pair_frequency/(letter_frequency[pair_word[0]]*letter_frequency[pair_word[1]])
#         return scores_return
#     def merge_pair_of_words(self, letter1,letter2,words_splits):

#         """
#         Merges a given pair of subwords (letter1, letter2) in all words' tokenized splits.

#         Process:
#         1. Iterates through each word and its tokenized split.
#         2. Checks if the word contains more than one token (i.e., can be merged).
#         3. Iterates through the word's tokenized split to find adjacent occurrences of (letter1, letter2).
#         4. Merges the pair based on whether letter2 starts with '##' (indicating it's a continuation).
#         5. Updates the tokenized representation after merging the pair.
#         6. Returns the modified word splits with merged subwords.

#         Parameters:
#         - letter1 (str): First token to merge.
#         - letter2 (str): Second token to merge (which follows letter1 in a split).
#         - words_splits (dict): Dictionary mapping words to their tokenized representation.

#         Returns:
#         - words_splits (dict): Updated dictionary with merged token pairs.
#         """

#         for word,word_split in words_splits.items():
#             if len(word_split)<=1:
#                 continue
#             else:
#                 word_split_modified=word_split
#                 ind_to_iterate=0
#                 while ind_to_iterate+1<len(word_split_modified):
#                     if word_split_modified[ind_to_iterate]==letter1 and word_split_modified[ind_to_iterate+1]==letter2:
#                         if letter2.startswith("##"):
#                             merged_letters=letter1+letter2[2:]
#                             word_split_modified=word_split_modified[:ind_to_iterate]+[merged_letters]+word_split_modified[ind_to_iterate+2:]
#                         else:
#                             merged_letters=letter1+letter2
#                             word_split_modified=word_split_modified[:ind_to_iterate]+[merged_letters]+word_split_modified[ind_to_iterate+2:]
#                     else:
#                         ind_to_iterate+=1
#                 words_splits[word]=word_split_modified
#         return words_splits
    

#     def construct_vocabulary(self):

#         """
#         Constructs a vocabulary using an iterative approach to merge subwords.

#         Process:
#         1. Extracts words and their tokenized splits from the corpus.
#         2. Initializes a vocabulary list with unique subwords from the splits.
#         3. Iteratively computes pair scores and merges the most frequent pair.
#         4. Updates the vocabulary list with newly merged tokens.
#         5. Stops when no further merging is possible.
#         6. Sorts and finalizes the vocabulary, adding special tokens.
#         7. Saves the vocabulary to a file.

#         Returns:
#         - vocabulary_return (list): The final constructed vocabulary.
#         """
#         words_frequency,words_split=self.extract_words()
#         vocabulary_return=[]
#         for words,word_split in words_split.items():
#             for letters in word_split:
#                 if letters not in vocabulary_return:
#                     vocabulary_return.append(letters)
        

#         for epoch in range(self.iterations):
#             scores_of_pairs=self.compute_pair_score(words_frequency,words_split)
#             best_pair_temp,max_score_temp=None,None
#             for pair_loop,score_loop in scores_of_pairs.items():
#                 if best_pair_temp==None or max_score_temp==None or max_score_temp<score_loop:
#                     best_pair_temp=pair_loop
#                     max_score_temp=score_loop
#             if best_pair_temp!=None and max_score_temp!=None:
#                 words_split=self.merge_pair_of_words(best_pair_temp[0],best_pair_temp[1],words_split)
#                 if best_pair_temp[1].startswith("##"):
#                     token_to_append=best_pair_temp[0]+best_pair_temp[1][2:]
#                     if token_to_append not in vocabulary_return:
#                         vocabulary_return.append(token_to_append)
#                 else:
#                     token_to_append=best_pair_temp[0]+best_pair_temp[1]
#                     if token_to_append not in vocabulary_return:
#                         vocabulary_return.append(token_to_append)
#             else:
#                 break
        
#         vocabulary_return.sort()
#         vocabulary_return=["[UNK]","[PAD]"]+vocabulary_return
#         self.vocabulary=vocabulary_return
#         with open(self.path_for_vocab_txt_out,"w") as file:
#             for tokens in self.vocabulary:
#                 file.write(tokens+"\n")
        
#         return self.vocabulary
    

#     def tokenize_helper(self,sentence):


#         """
#         Tokenizes a given sentence using a WordPiece-based approach.

#         Process:
#         1. Extracts all lowercase words from the sentence.
#         2. Iterates over each word to find the longest matching subword in the vocabulary.
#         3. If a match is found, it is added to the tokenized output.
#         4. If no match is found, the [UNK] token is used as a fallback.
#         5. Returns a list of tokens representing the input sentence.

#         Args:
#         - sentence (str): The input sentence to be tokenized.

#         Returns:
#         - token_ret (list): A list of tokens representing the tokenized sentence.
#         """
#         words_in_the_sentence=re.findall(r'\b[a-z]+\b', sentence)
#         token_ret=[]


#         for word in words_in_the_sentence:
                
#                 tokens_word=[]


#                 start_ind_of_word=0
#                 while(start_ind_of_word<len(word)):

#                     found_bool=False
#                     for end_ind_of_word in range(len(word),start_ind_of_word,-1):


#                         if start_ind_of_word==0:
#                             subword_token_to_find=word[start_ind_of_word:end_ind_of_word]

                        

#                             if subword_token_to_find in self.vocabulary:
#                                 token_ret.append(subword_token_to_find)
#                                 start_ind_of_word=end_ind_of_word
#                                 found_bool=True
#                                 break
#                         else:
#                             subword_token_to_find="##"+word[start_ind_of_word:end_ind_of_word]

#                             if subword_token_to_find in self.vocabulary:
#                                 token_ret.append(subword_token_to_find)
#                                 start_ind_of_word=end_ind_of_word
#                                 found_bool=True
#                                 break

#                     if found_bool==False:
#                         tokens_word=[self.vocabulary[0]]
#                         break
                    
#                     token_ret=token_ret+tokens_word
#         return token_ret

#     def tokenize(self,sentences_in_json,token_out_json):

#         """
#         Tokenizes sentences from a JSON file and saves the output in another JSON file.

#         Process:
#         - Reads a JSON file containing sentences.
#         - Tokenizes each sentence using the `tokenize_helper` method.
#         - Stores the tokenized sentences in a dictionary.
#         - Writes the output dictionary to a new JSON file.
        
#         Args:
#         sentences_in_json (str): Path to the input JSON file containing sentences.
#         token_out_json (str): Path to the output JSON file to save tokenized sentences.

#         Returns:
#         dict: A dictionary with sentence IDs as keys and tokenized words as values.
#         """
#         with open(sentences_in_json,"r") as f:
#             data_in=json.load(f)
#         data_out={}

#         for entries in data_in:
#             id_of_sen=entries["id"]
#             sentence_of_id=entries["sentence"]
#             data_out[id_of_sen]=self.tokenize_helper(sentence_of_id)
        
#         with open(token_out_json,"w") as f:
#             json.dump(data_out,f,indent=4)
        
#         print(data_out)
#         return data_out

        

                    

