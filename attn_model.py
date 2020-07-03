from torch.nn import utils as nn_utils
import torch
from torch import optim
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.init as init
class SimpleAttnClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, label_dim=1, scale=10, attn_type='dot'):
        super(SimpleAttnClassifier, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.embeddings.weight.data.uniform_(-0.1, 0.1)
        self.dropout = nn.Dropout(0.5)

        self.affine = nn.Linear(embed_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.attn_linear = nn.Linear(hidden_dim, hidden_dim)
        self.scale = scale
        self.V = nn.Parameter(torch.randn(hidden_dim, 1))
        self.decoder = nn.Linear(hidden_dim, label_dim, bias=False)
        self.attn_type = attn_type

    # batch_size * sent_l * dim
    def forward(self, seq_ids, seq_lengths=None):
        '''
        Args:
            seq_ids: word indexes, batch_size, max_len, Long Tensor
            seq_lengths: lengths of sentences, batch_size, Long Tensor
        attention:
            score = v h
            att = softmax(score)
        '''
        
        seq_embs = self.embeddings(seq_ids)
        seq_embs = self.dropout(seq_embs)
        batch_size, max_len, hidden_dim = seq_embs.size()
        # batch * max_len * hidden_states
        #hidden_vecs = self.affine(seq_embs)
        hidden_vecs = seq_embs
        if self.attn_type == 'dot':
            inter_out = hidden_vecs
        else:
            inter_out = torch.tanh(self.attn_linear(hidden_vecs))
        #batch * max_len
        scores = torch.matmul(inter_out, self.V).squeeze(-1)
        scores = scores/self.scale
        #Mask the padding values
        mask = torch.zeros_like(seq_ids)
        for i in range(batch_size):
            mask[i, seq_lengths[i]:] = 1
        scores = scores.masked_fill(mask.bool(), -np.inf)
        #Softmax, batch_size*1*max_len
        attn = self.softmax(scores).unsqueeze(1)
        #weighted sum, batch_size*hidden_dim
        final_vec = torch.bmm(attn, hidden_vecs).squeeze(1)
        final_vec = self.dropout(final_vec)
        senti_scores = self.decoder(final_vec)
        probs = self.sigmoid(senti_scores)
        return probs, senti_scores, attn, final_vec
    
    def load_vector(self, path, trainable=False):
        '''
        Load pre-savedd word embeddings
        '''
        with open(path, 'rb') as f:
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(path, vectors.shape))
            #self.word_embed.weight = nn.Parameter(torch.FloatTensor(vectors))
            self.embeddings.weight.data.copy_(torch.from_numpy(vectors))
            self.embeddings.weight.requires_grad = trainable
            print('embeddings loaded')