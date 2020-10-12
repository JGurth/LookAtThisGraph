import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (9, 7)
mpl.rcParams['font.size'] = 15
import numpy as np
import torch
from lookatthisgraph.utils.dataset import Dataset
from lookatthisgraph.utils.trainer import Trainer
from lookatthisgraph.utils.model import Model
from datetime import datetime, timedelta

FileLocation="Data/140000"
train_set = Dataset([FileLocation])

width=128
conv_depth=3
lin_depth=5
k_max=2    #=k von K-fold validation
k_size=int(2e3)  #=size of K-fold sample

resultlist=[]

#True/False config statement
#if true scikit split/kfold
#else normal
#configvalue k_max
#configvalue k_crnt
#iteriere k_crnt in example
#k_crnt legt testingpart fest.


for k_crnt in range(k_max):
    train_config = {
            'learning_rate': 7e-4,
            'scheduling_step_size': 30,        
            'scheduling_gamma': .7,
            'training_target': 'energy',
            'dim': [width, conv_depth, lin_depth],
            'dataset': train_set,
            'train_split': 1e5,
            'test_split': 2e3,
            'batch_size': 512,
            'max_epochs': 100,
            'kFold_max' : k_max,
            'kFold_crnt' : k_crnt,
            'kFold_size' : k_size
        }
                
    trainer = Trainer(train_config)
    trainer.train()
    trainer.load_best_model()
    prediction, truth = trainer.evaluate_test_samples()
    
    prediction=torch.from_numpy(prediction)
    truth=torch.from_numpy(truth[train_config['training_target']].flatten())
    avrg=torch.mean(torch.square(torch.sub(prediction, truth))).item()  #Average
    result=torch.tensor([avrg])
    resultlist.append(result)
    
    endresult=torch.cat(resultlist, 0)  

endresult=torch.mean(endresult).item()
print('k-Fold final Accuracy:', endresult)
