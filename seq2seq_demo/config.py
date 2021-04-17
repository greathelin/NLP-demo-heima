from num_sequence import NumSequence

num_sequence = NumSequence()
train_batch_size = 1024
max_len = 9

embedding_dim = 100
num_layer = 1
hidden_size = 64

model_save_path = "model/seq2seq.pth"
optimizer_save_path = "model/optimizer.pth"

device = 'cuda'
