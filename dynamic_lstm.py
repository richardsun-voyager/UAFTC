from torch.nn import utils as nn_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.init as init
def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)
            
class dynamicLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers=1, bidirectional=False, dropout=0):
        super(dynamicLSTM, self).__init__()

        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers = num_layers,
            bidirectional=bidirectional, dropout=dropout)
        init_ortho(self.rnn)

    # batch_size * sent_l * dim
    def forward(self, feats, seq_lengths=None):
        '''
        Args:
        feats: batch_size, max_len, emb_dim
        seq_lengths: batch_size
        '''
        perm_seq_lens, perm_idx = seq_lengths.sort(0, descending=True)
        _, desorted_perm_idx = torch.sort(perm_idx, descending=False)
        perm_seqs = feats[perm_idx]
        
        pack = nn_utils.rnn.pack_padded_sequence(perm_seqs, 
                                                 perm_seq_lens, batch_first=True)
        
        #batch_size*max_len*hidden_dim
        self.rnn.flatten_parameters()
        rnn_out, (final_output, _) = self.rnn(pack)
        #batch_size, hidden_dim*bidirectional
        final_output = final_output.transpose(0, 1)
        final_output = final_output[desorted_perm_idx].view(len(feats), -1)
        #Unpack the tensor, get the output for varied-size sentences
        #padding with zeros
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        # batch * sent_l * 2 * hidden_states 
        return unpacked[desorted_perm_idx], final_output