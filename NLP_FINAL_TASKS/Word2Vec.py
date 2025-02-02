import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from WordTokenizer import WordPieceTokenizer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
import random
class Word2VecDataset(Dataset):
    def __init__(self,text, WordPieceTokenizer:WordPieceTokenizer,context_size=2):
        self.tokenizer=WordPieceTokenizer
        self.vocabulary=self.tokenizer.get_vocabulary()
        self.token_word_to_dict={}
        self.context_size=context_size
        for i in range(len(self.vocabulary)):
            self.token_word_to_dict[self.vocabulary[i]]=i
        self.data=self.pre_process_data(text)
    def pre_process_data(self,text_file):
        with open(text_file,"r") as f:
            sentences=[line.strip() for line in f.readlines() if line.strip()]
        data=[]
        for sentence in sentences:
            list_of_tokens_for_sentence=self.tokenizer.tokenize_helper(sentence)
            
            for i in range(len(list_of_tokens_for_sentence)):
                list_of_context=[]
                
                for j in range(i-self.context_size,i+self.context_size+1):

                    if j<0 or j>=len(list_of_tokens_for_sentence):
                        list_of_context.append(1)
                        continue
                    if j==i:
                        continue
                    list_of_context.append(self.token_word_to_dict[list_of_tokens_for_sentence[j]])

                list_of_context_tensor=torch.tensor(list_of_context, dtype=torch.long)

                target_tensor=torch.tensor(self.token_word_to_dict[list_of_tokens_for_sentence[i]], dtype=torch.long)
                data.append((list_of_context_tensor,target_tensor))
        return data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]



# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dimension):
        super(Word2VecModel, self).__init__()
        self.embedd = nn.Embedding(vocab_size, embedding_dimension)
        self.linear_layer = nn.Linear(embedding_dimension, vocab_size)
        self.trainlosshis = []
        self.vallosshis = []
        self.to(device)  # Move model to GPU

    def forward(self, context_):
        context_ = context_.to(device)  # Ensure input is on GPU
        embed_temp = self.embedd(context_).mean(dim=1)
        out_put_tex = self.linear_layer(embed_temp)
        return out_put_tex

    def train(self, train_loader, val_loader, epochs, learning_rate,optimizer="SGD"):
        lossfunction = nn.CrossEntropyLoss()
        self.opt=None
        if optimizer=="Adam":
            self.opt = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer=="SGD":
            self.opt = optim.SGD(self.parameters(), lr=learning_rate)

        self.epochs=epochs
        for epoch in range(1,epochs+1):
            # Training phase
            trainlosstot = 0
            for context, target in train_loader:
                context = context.to(device)
                target=target.to(device)  # Move data to GPU

                self.zero_grad()
                outtemp = self(context)
                losstemp = lossfunction(outtemp, target)
                losstemp.backward()
                self.opt.step()
                trainlosstot += losstemp.item()
            
            avgtrain = trainlosstot / len(train_loader)
            self.trainlosshis.append(avgtrain)

            # Validation phase
            vallosstot = 0
            with torch.no_grad():  # No gradient calculation for validation
                for context, target in val_loader:
                    context = context.to(device)
                    target= target.to(device)
                    outtemp = self(context)
                    losstemp = lossfunction(outtemp, target)
                    vallosstot += losstemp.item()
            
            avgval = vallosstot / len(val_loader)
            self.vallosshis.append(avgval)
            print(f"For epoch --> {epoch}/{epochs}, training Loss is {avgtrain} and validation loss is {avgval}")

    def save_the_model(self,checkpoint_path):
        checkpoint_to_save = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "trainlosshistory": self.trainlosshis,
            "vallosshistory": self.vallosshis,
            "epochs": self.epochs + 1
        }

        torch.save(checkpoint_to_save, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")


    def plot_val_loss_vs_training_loss(self):
        plt.plot(range(1, self.epochs + 1), self.trainlosshis, label="Training Loss")
        plt.plot(range(1, self.epochs + 1), self.vallosshis, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.show()

    def compute_cosine_similarity(self):
        embedd_words=self.embedd.weight.data
        magnitude_of_vectors=torch.norm(embedd_words,p=2,dim=1,keepdim=True)
        norm_embed=embedd_words/magnitude_of_vectors
        sim_matrix=torch.mm(norm_embed,norm_embed.T)
        return sim_matrix

        
    def make_one_triplet(self,index,vocabulary,similairty_matrix):
        similarity_for_i_word=similairty_matrix[index]
        _,sim_indices=torch.topk(similarity_for_i_word,2)
        sim_indices_integer=sim_indices[1].item()
        dim_sim_ind=torch.argmin(similarity_for_i_word)
        dim_sim_index_int=dim_sim_ind.item()
        return (vocabulary[index],vocabulary[sim_indices_integer],vocabulary[dim_sim_index_int])

    def two_sim_one_disim_triplets(self,vocabulary):
        sim_matrix=self.compute_cosine_similarity()
        two_int=random.sample(range(len(vocabulary)),2)
        triplets_ret=[self.make_one_triplet(two_int[0],vocabulary,sim_matrix),self.make_one_triplet(two_int[1],vocabulary,sim_matrix)]
        return triplets_ret





test_class=WordPieceTokenizer("corpus.txt","vocabulary_out.txt",100000,0)
test_class.construct_vocabulary()





dataset = Word2VecDataset("corpus.txt", test_class,2)

train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32  # Adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Print dataset sizes
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(dataset.token_word_to_dict)



vocab_size = len(test_class.get_vocabulary()) # Replace with actual vocabulary size
print(vocab_size)
embedding_dim = 100
model=Word2VecModel(vocab_size,embedding_dim)
print(model)
model.train(train_loader,val_loader,100,0.01)
model.save_the_model("Word2VecCheckpoint.pth")
model.plot_val_loss_vs_training_loss()







vocab_size = len(test_class.get_vocabulary()) # Replace with actual vocabulary size
print(vocab_size)
embedding_dim = 100
model1=Word2VecModel(vocab_size,embedding_dim)
print(model)
model1.train(train_loader,val_loader,100,0.01,"Adam")
model1.save_the_model("Word2VecCheckpoint.pth")
model1.plot_val_loss_vs_training_loss()