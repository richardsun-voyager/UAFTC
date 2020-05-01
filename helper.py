import pickle
import numpy as np
import os
import copy
import torch

class data_generator:
    def __init__(self, config, data_path, is_training=True):
        '''
        Generate training and testing samples
        Args:
        config: configuration parameters
        data_path: data path, string
        data_batch: data list, each contain a nametuple
        '''    
        self.is_training = is_training
        self.config = config
        self.index = 0
        #Filter sentences without targets
        self.data_backup = list(self.load_data(data_path))
        self.data_batch = copy.deepcopy(self.data_backup)
        self.data_len = len(self.data_batch)
        self.UNK = "unk"
        self.EOS = "<eos>"
        self.PAD = "<pad>"
        self.load_local_dict()
        
#         options_file = config.elmo_config_file 
#         weight_file = config.elmo_weight_file
#         if options_file != '':
#             self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
            
    def load_data(self, path):
        '''
        Load the pickle file
        '''
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data


    def load_local_dict(self):
        '''
        Load dictionary files
        '''
        if not os.path.exists(self.config.dic_path):
            print('Dictionary file not exist!')
        with open(self.config.dic_path, 'rb') as f:
            vocab, word2id, id2word = pickle.load(f)
        self.UNK_ID = word2id[self.UNK]
        self.PAD_ID = word2id[self.PAD]
        self.EOS_ID = word2id[self.EOS]

    def generate_sample(self, data):
        '''
        Generate a batch of training dataset
        '''
        batch_size = self.config.batch_size
        select_index = np.random.choice(len(data), batch_size, replace=False)
        select_data = [data[i] for i in select_index]
        return select_data

    def generate_balanced_sample(self, data):
        '''
        Generate balanced training data set 
        rate: list, i.e., [0.6, 0.2, 0.2]
        '''
        # np.random.seed(222)
        batch_size = self.config.batch_size
        #labels must be number in order to sort
        labels = [item[2] for item in all_triples]
        unique_label, count_label = np.unique(labels, return_counts=True)
        rate = 1.0/count_label
        p = [rate[item[2]] for item in data]
        p = p/sum(p)
        select_index = np.random.choice(len(data), batch_size, p=p, replace=False)
        select_data = [data[i] for i in select_index]
        return select_data

    def elmo_transform(self, data):
        '''
        Transform sentences into elmo, each sentence represented by words
        '''
        token_list, mask_list, label_list, _, texts, targets, _ = zip(*data)
        sent_lens = [len(tokens) for tokens in token_list]
        sent_lens = torch.LongTensor(sent_lens)
        label_list = torch.LongTensor(label_list)
        max_len = max(sent_lens)
        batch_size = len(sent_lens)
        character_ids = batch_to_ids(token_list)
        embeddings = self.elmo(character_ids)
        
        #batch_size*word_num * 1024
        sent_vecs = embeddings['elmo_representations'][0]
        sent_vecs = sent_vecs.detach()#no gradient
        #Padding the mask to same lengths
        mask_vecs = np.zeros([batch_size, max_len])
        mask_vecs = torch.LongTensor(mask_vecs)
        for i, mask in enumerate(mask_list):
            mask_vecs[i, :len(mask)] = torch.LongTensor(mask)
        return sent_vecs, mask_vecs, label_list, sent_lens, texts, targets

    def shuffle_data(self):
        np.random.shuffle(self.data_batch)
#         if epoch_index:
#             #np.random.seed(10)
#             #shuffled_index = np.random.permutation(self.data_len)
#             #self.data_batch = np.array(self.data_backup)[shuffled_index]
#             np.random.shuffle(self.data_batch)
#         else:
#             np.random.shuffle(self.data_batch)

    def reset_samples(self):
        self.index = 0

    def pad_data(self, sents, labels):
        '''
        Padding sentences to same size
        '''
        sent_lens = [len(tokens) for tokens in sents]
        sent_lens = torch.LongTensor(sent_lens)
        label_list = torch.LongTensor(labels)
        max_len = max(sent_lens)
        batch_size = len(sent_lens)

        #padding sent with PAD IDs
        sent_vecs = np.ones([batch_size, max_len]) * self.PAD_ID
        sent_vecs = torch.LongTensor(sent_vecs)
        for i, s in enumerate(sents):#batch_size*max_len
            sent_vecs[i, :len(s)] = torch.LongTensor(s)
        sent_lens, perm_idx = sent_lens.sort(0, descending=True)
        sent_ids = sent_vecs[perm_idx]

        label_list = label_list[perm_idx]

        return sent_ids, label_list, sent_lens
            
    def get_ids_samples(self, is_balanced=False):
        '''
        Get samples including ids of words, labels
        '''
        if self.is_training:
            if is_balanced:
                samples = self.generate_balanced_sample(self.data_batch)
            else:
                samples = self.generate_sample(self.data_batch)
            token_ids, label_list = zip(*samples)
            #Sorted according to the length
            sent_ids, label_list, sent_lens = self.pad_data(token_ids, label_list)
        else:
            if self.index == self.data_len:
                print('Testing Over!')
            #First get batches of testing data
            if self.data_len - self.index >= self.config.batch_size:
                #print('Testing Sample Index:', self.index)
                start = self.index
                end = start + self.config.batch_size
                samples = self.data_batch[start: end]
                self.index = end
                token_ids, label_list = zip(*samples)
                #Sorting happens here
                sent_ids,  label_list, sent_lens = self.pad_data(token_ids, label_list)

            else:#Then generate testing data one by one
                samples =  self.data_batch[self.index:] 
                if self.index == self.data_len - 1:#if only one sample left
                    samples = [samples]
                token_ids, label_list = zip(*samples)
                sent_ids,  label_list, sent_lens = self.pad_data(token_ids, label_list)
                self.index += len(samples)
        yield sent_ids,  label_list, sent_lens


    def get_sequential_ids_samples(self, is_balanced=False):
        '''
        Get samples including ids of words, labels
        '''

        if self.index == self.data_len:
            print('Testing Over!')
        #First get batches of testing data
        if self.data_len - self.index >= self.config.batch_size:
            #print('Testing Sample Index:', self.index)
            start = self.index
            end = start + self.config.batch_size
            samples = self.data_batch[start: end]
            self.index = end
            token_ids, label_list = zip(*samples)
            #Sorting happens here
            sent_ids, label_list, sent_lens = self.pad_data(token_ids, label_list)

        else:#Then generate testing data one by one
            samples =  self.data_batch[self.index:]
            if self.index == self.data_len - 1:#if only one sample left
                samples = [samples]
            token_ids, label_list = zip(*samples)
            sent_ids, label_list, sent_lens = self.pad_data(token_ids, label_list)
            self.index += len(samples)
        yield sent_ids, label_list, sent_lens



    def get_elmo_samples(self, is_with_texts=False):
        '''
        Generate random samples for training process
        Generate samples for testing process
        sentences represented in Elmo
        '''
        if self.is_training:
            samples = self.generate_sample(self.data_batch)
            sent_vecs, mask_vecs, label_list, sent_lens, texts, targets = self.elmo_transform(samples)
            #Sort the lengths, and change orders accordingly
            sent_lens, perm_idx = sent_lens.sort(0, descending=True)
            sent_vecs = sent_vecs[perm_idx]
            mask_vecs = mask_vecs[perm_idx]
            label_list = label_list[perm_idx]
            texts = [texts[i.item()] for i in perm_idx]
            targets = [targets[i.item()] for i in perm_idx]
        else:
            if self.index == self.data_len:
                print('Testing Over!')
            #First get batches of testing data
            if self.data_len - self.index >= self.config.batch_size:
                #print('Testing Sample Index:', self.index)
                start = self.index
                end = start + self.config.batch_size
                samples = self.data_batch[start: end]
                self.index += self.config.batch_size
                sent_vecs, mask_vecs, label_list, sent_lens, texts, targets = self.elmo_transform(samples)
                #Sort the lengths, and change orders accordingly
                sent_lens, perm_idx = sent_lens.sort(0, descending=True)
                sent_vecs = sent_vecs[perm_idx]
                mask_vecs = mask_vecs[perm_idx]
                label_list = label_list[perm_idx]
                texts = [texts[i.item()] for i in perm_idx]
                targets = [targets[i.item()] for i in perm_idx]
            else:#Then generate testing data one by one
                samples =  self.data_batch[self.index] 
                sent_vecs, mask_vecs, label_list, sent_lens, texts, targets = self.elmo_transform([samples])
                self.index += 1
        if is_with_texts:
            yield sent_vecs, mask_vecs, label_list, sent_lens, texts, targets
        else:
            yield sent_vecs, mask_vecs, label_list, sent_lens