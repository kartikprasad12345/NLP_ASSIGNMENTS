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

        

                    


