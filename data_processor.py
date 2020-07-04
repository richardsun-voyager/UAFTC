import pandas as pd
import numpy as np
import pickle
import en_core_web_sm
import itertools
import io
from collections import Counter
from multiprocessing import Pool#multi-threads
from functools import partial
nlp = en_core_web_sm.load()

#The csv files have two columns, 
#one is the text, the other is the label
path1 = 'data/train.csv'
path2 = 'data/dev.csv'
path3 = 'data/test.csv'

root = 'data/'
glove_path = 'glove/glove.840B.300d.txt'
pretrained_word_emb = {}
def tokenize(text):
    return [w.text.lower() for w in nlp.tokenizer(text)]

def load_glove_word_emb(file_path, embed_dim=300):
    '''
    Load a specified vocabulary
    '''
    word_emb = {}
    vocab_words = set()
    with open(file_path) as fi:
        for line in fi:
            items = line.split()
            word = ' '.join(items[:-1*embed_dim])
            vec = items[-1*embed_dim:]
            word_emb[word] = np.array(vec, dtype=np.float32)
            vocab_words.add(word)
    return word_emb, vocab_words

def load_fasttext_word_emb(file_path, embed_dim=300):
    '''
    Load a specified vocabulary
    '''
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    word_emb = {}
    words = []
    for line in fin:
        tokens = line.rstrip().split(' ')
        words.append(tokens[0])
        word_emb[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
    return word_emb, words

def word2vec(word):
    '''
    Map a word into a vec
    '''
    try:
        vec = pretrained_word_emb[word]
    except:
        vec = pretrained_word_emb['unk']
    return vec

def word_id(w, word2id):
    '''
    Map a word into an id in the vocabulary
    '''
    try:
        index = word2id[w]
    except:
        index = word2id['unk']
    return index

def seq_ids(sequence, word2id):
    '''
    Map a text into a sequence of ids
    '''
    word_ids = [word_id(w, word2id) for w in sequence]
    return word_ids

def main():
    #Load the data
    train_data = pd.read_csv(path1)
    dev_data = pd.read_csv(path2)
    test_data = pd.read_csv(path3)
    #Get all the tokens
    with Pool(12) as pool:
        print('Start tokenization')
        text3 = list(test_data.text.values)
        text_words3 = list(pool.map(tokenize, text3))
        print('Go on tokenization')
        text1 = list(train_data.text.values)
        text_words1 = list(pool.map(tokenize, text1))
        text2 = list(dev_data.text.values)
        text_words2 = list(pool.map(tokenize, text2))
    
    #Merge tokens in the training, valid and test set
    total_words = text_words1 + text_words2 + text_words3
    #del text1, text2, text3
    #del test_data, train_data, dev_data
    total_words = list(itertools.chain(*total_words))
    #Build a vocabulary
    word_freq = Counter(total_words)
    max_word_num = 300000
    if len(list(word_freq.keys())) < 300000:
        max_word_num = len(list(word_freq.keys()))
        print('Vocabulary size: ', max_word_num)
    common_words = word_freq.most_common(max_word_num)
    vocab, _ = zip(*common_words)
    vocab =list(vocab)
    if 'unk' not in vocab:
        vocab = vocab + ['unk']
        vocab = vocab + ['<eos>']
        vocab = vocab + ['<pad>']
    #Build a dictionary for mapping words to ids
    word2id= {w:i for i, w in enumerate(vocab)}
    id2word = {i:w for i, w in enumerate(vocab)}
    #Save the dictionary as a binary format
    with open(root+'vocab/local_dict.pkl', 'wb') as f:
        pickle.dump([vocab, word2id, id2word], f)
        print('Vocabulary saved successfully')
        del word_freq
        #del text_words1, text_words2, text_words3
        
    #Get the corresponding embedding for each token
    word_emb, _ = load_glove_word_emb(glove_path)
    global pretrained_word_emb
    pretrained_word_emb = word_emb
    #print(pretrained_word_emb['unk'])
    
    ##pool.map(partial(func, b=second_arg), a_args)
    print('Start to find embeddings')
    with Pool(12) as p:
        selected_emb = list(p.map(word2vec, vocab))
    #selected_index = [find_index(w, pre_trained_vocab_words) for w in vocab]
    selected_emb = np.vstack(selected_emb)
    #Save the embeddings
    with open(root+'vocab/local_emb.pkl', 'wb') as f:
        pickle.dump(selected_emb, f)
        print('Local embeddings saved successfully')
    
    #Map words to indexes
    print('Mapping words to indexes')
    with Pool(6) as p:
        train_seq_ids = list(p.map(partial(seq_ids, word2id=word2id), text_words1))
        dev_seq_ids = list(p.map(partial(seq_ids, word2id=word2id), text_words2))
        test_seq_ids = list(p.map(partial(seq_ids, word2id=word2id), text_words3))
        
    train_seq = zip(train_seq_ids, train_data.label.values)
    dev_seq = zip(dev_seq_ids, dev_data.label.values)
    test_seq = zip(test_seq_ids, test_data.label.values)
    print('Mapping over')
    print('Save them to local files')
    with open(root+'train_seq.pkl', 'wb') as f:
        pickle.dump(train_seq, f)
    with open(root+'dev_seq_.pkl', 'wb') as f:
        pickle.dump(dev_seq, f)    
    with open(root+'test_seq.pkl', 'wb') as f:
        pickle.dump(test_seq, f)  
    

if __name__ == "__main__":
    main()