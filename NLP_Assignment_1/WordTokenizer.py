import json
from collections import defaultdict
class WordPieceTokenizer:
    def __init__(self,corpus_file_path,path_for_vocab_txt_out,sentences_id_in_json,token_id_out_json,iterations,max_vocab_size):
        self.corpus_file_path=corpus_file_path
        self.iterations=iterations
        self.vocabulary=None
        self.path_for_vocab_txt_out=path_for_vocab_txt_out
        self.max_vocab_size=max_vocab_size
        self.sentences_id_in_json=sentences_id_in_json
        self.token_id_out_json=token_id_out_json
        with open(corpus_file_path, 'r') as file:
            self.corpus=file.read()
        self.corpus=self.corpus
    def get_vocabulary(self):
        return self.vocabulary
    def extract_words(self):
        words_freq={}
        word=""
        for character in self.corpus:
            if 'a'<=character and character<='z':
                word+=character
            elif( character>= "!" and character<= "/") or( character>= ":" and character<="?"):
                if character not in words_freq:
                    words_freq[character]=1
                      
                else:
                    words_freq[character]+=1
                   

                if word  != "":
                    if word not in words_freq:
                        words_freq[word]=1
                        word=""
                    else:
                        words_freq[word]+=1
                        word=""
            else:
                if word  != "":
                    if word not in words_freq:
                        words_freq[word]=1
                        word=""
                    else:
                        words_freq[word]+=1
                        word=""
        if word != "":
            if word not in words_freq:
                words_freq[word]=1
            else:
                words_freq[word]+= 1


        print(words_freq)

        word_splits={}
        for i in words_freq.keys():
            word_temp=[]
            o=True
            for k in i:
                if o:
                    word_temp.append(k)
                    o=False
                else:
                    word_temp.append("##"+k)
            word_splits[i]=word_temp


        print(word_splits,words_freq)
        return word_splits,words_freq
    def init_vocab(self, word_split):
        vocab_init=["[PAD]","[UNK]"]
        for i,j in word_split.items():
            for k in j:
                if k not in vocab_init:
                    vocab_init.append(k)
        return vocab_init
    def compute_pair_score(self, words_freq, word_splits):
        pair_freq=defaultdict(int)
        letter_freq=defaultdict(int)
        for word,freq in words_freq.items():
            split_of_word=word_splits[word]
            if(len(split_of_word)==1):
                letter_freq[split_of_word[0]]+=1
            else:
                for i in range(len(split_of_word)-1):
                    pair_of_split=(split_of_word[i],split_of_word[i+1])

                    letter_freq[split_of_word[i]]+=freq

                    pair_freq[pair_of_split]+=freq

                letter_freq[split_of_word[-1]]+=freq
        scores={}
        for pair_word,freq in pair_freq.items():
            scores[pair_word]=freq/(letter_freq[pair_word[0]]*letter_freq[pair_word[1]])
        return scores
    def merge_pair(self,a,b, splits):
        for word_temp, split_temp in splits.items():
            split_temp_loop=split_temp
            if len(split_temp_loop)>1:
                i=0
                while i<len(split_temp_loop)-1:
                    if split_temp_loop[i]==a and split_temp_loop[i+1]==b:
                        merge=""
                        if b.startswith("##"):
                            merge=a+b[2:]
                        else:
                            merge=a+b
                        split_temp_loop=split_temp_loop[:i]+[merge]+split_temp_loop[i+2:]
                    else:
                        i+=1
                splits[word_temp]=split_temp_loop
        return splits
    def construct_vocabulary(self):
        word_splits,words_freq=self.extract_words()
        vocab_temp=self.init_vocab(word_splits)
        for i in range(self.iterations):
            score_temp=self.compute_pair_score(words_freq,word_splits)
            best_pair=None
            max_score=None
            for pair_temp,prob_temp in score_temp.items():
                if best_pair is None or max_score is None or max_score<prob_temp:
                    best_pair=pair_temp
                    max_score=prob_temp
            if best_pair==None or max_score==None:
                break
            word_splits=self.merge_pair(best_pair[0],best_pair[1],word_splits)
            
            token_to_insert=""
            if(best_pair[1][0:2]=="##"):
                token_to_insert=best_pair[0]+best_pair[1][2:]
            else:
                token_to_insert=best_pair[0]+best_pair[1]
            if token_to_insert not in vocab_temp:
                vocab_temp.append(token_to_insert)
        print("__________")
        # self.vocabulary.sort()
        self.vocabulary=vocab_temp
        self.vocabulary.sort()

        self.write_vocab_to_file()
        return vocab_temp
    def write_vocab_to_file(self):
        with open(self.path_for_vocab_txt_out,"w") as file:
            for tokens in self.vocabulary:
                file.write(tokens+"\n")


    
    def extract_from_json(self,path_to_json):
        with open(path_to_json,"r") as f:
            test_data=json.load(f)
        return test_data



    def tokenize_word(self,word_to_tokenize):
        tokens_res=[]
        start_ind=0
        while start_ind<len(word_to_tokenize):
            end_ind=len(word_to_tokenize)

            found_token=False

            while start_ind<end_ind:
                subword_to_find=word_to_tokenize[start_ind:end_ind]
                if start_ind>0:
                    subword_to_find="##"+subword_to_find

                if subword_to_find in self.vocabulary:
                    tokens_res.append(subword_to_find)
                    start_ind=end_ind
                    found_token=True
                    break
                end_ind-=1
            if not found_token:
                tokens_res.append("[UNK]")
                start_ind+=1
        return tokens_res

    def tokenize_sentence(self, text_to_tokenize):
        sentence_to_tokenize=[]
        word=""
        for character in text_to_tokenize:
            if 'a'<=character and character<='z':
                word+=character
            else:
                if word  != "":
                    sentence_to_tokenize.append(word)
                    word=""
                    
        if word != "":
            sentence_to_tokenize.append(word)
        res_tokens_divided=[]
        for word_to_tokenize in sentence_to_tokenize:
            temp=[]
            temp=self.tokenize_word(word_to_tokenize)
            res_tokens_divided=res_tokens_divided+temp
        return res_tokens_divided



    def tokenize(self):
        test_data=self.extract_from_json(self.sentences_id_in_json)
        data_to_be_written={}
        for entry in test_data:
            id_sen=entry["id"]
            sen=entry["sentence"]
            data_to_be_written[id_sen]=self.tokenize_sentence(sen[:-1])


    
        out_file=self.token_id_out_json
        with open(out_file,"w") as f:
            json.dump(data_to_be_written,f,indent=4)
    