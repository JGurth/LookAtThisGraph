from torch_geometric.data import Dataset
from lookatthisgraph.utils.dataset import Dataset as DS
import logging
import numpy as np


class LDataset(DS, Dataset):
    'Needs config: train/test/validation_split and batch_size. Also needs to be used with LTrainer.'
    def __init__(self, file_list, config, normalization_parameters=None, logging_level=logging.INFO):
        super(LDataset, self).__init__(file_list, normalization_parameters, logging_level)
        
        self.config=config
        self.__indices__=None
        self.transform=None
        self.pre_transform=None
        self.permutation = np.random.permutation(len(self.data_list))
        self._train_split = config['train_split'] if 'train_split' in config else None
        self._test_split = config['test_split'] if 'test_split' in config else None
        self._val_split = self.config['validation_split'] if 'validation_split' in self.config else 'batch'
        self._batch_size = config['batch_size']
        
        
        split = lambda s: int(self.n_events * s) if s < 1 else int(s)
        if 'kFold_max' in self.config and 'kFold_crnt' in self.config:
            
            if 'kFold_size' in self.config:
                k_datalist=self.data_list[:self.config['kFold_size']]
            else:
                k_datalist=self.data_list
            
            
            size=len(k_datalist)//self.config['kFold_max'] #size of the k groups of data
            vallist=[]
            k_train=[]
            #Validation Size:
            n_val=self.config['validation_split'] if 'validation_split' in self.config else self.config['batch_size']
            #list of the k groups of Data
            for i in range(self.config['kFold_max']):
                vallist.append(k_datalist[i*size:(i+1)*size])
            #picking of the test group and recombination of the training group
            for i in range(len(vallist)):
                if i==self.config['kFold_crnt']:
                    k_test=vallist[i]
                else:
                    k_train+=vallist[i]
            #validiation group
            k_val=[self.data_list[i] for i in self.permutation][:n_val]
            
            n_test=len(k_test)
            n_train=len(k_train)
            
            logging.info('%d training samples, %d validation samples, %d test samples received; %d ununsed',
                    n_train, n_val, n_test, len(self.data_list) - n_train - n_val - n_test)
            if n_train + n_val + n_test > self.n_events:
                raise ValueError('Loader configuration exceeds number of data samples')
            
            self.train_list=k_train
            self.val_list=k_val
            self.test_list=k_test
        else:
            dataset_shuffled = [self.data_list[i] for i in self.permutation]
            if self._val_split == 'batch':
                n_val = self._batch_size
            else:
                n_val = split(self._val_split)
            if self._train_split is None:
                if self._test_split is not None:
                    n_test = split(self._test_split)
                else:
                    n_test = 0
                n_train = len(self.data_list) - n_val - n_test
            else:
                n_train = split(self._train_split)
                if self._test_split is not None:
                    n_test = split(self._test_split)
                else:
                    n_test = len(self.data_list) - n_train - n_val
            logging.info('%d training samples, %d validation samples, %d test samples received; %d ununsed',
                    n_train, n_val, n_test, len(self.data_list) - n_train - n_val - n_test)
            self.train_list=dataset_shuffled[:n_train]
            self.val_list=dataset_shuffled[n_train:n_train+n_val]
            self.test_list=dataset_shuffled[n_train+n_val:][:n_test]

    def len(self):
        return len(self.train_list)
    
    
    def get(self, idx):
        data=self.train_list[idx]
        return data
    
