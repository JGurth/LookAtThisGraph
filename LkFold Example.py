import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (9, 7)
mpl.rcParams['font.size'] = 15
import numpy as np
import torch
from lookatthisgraph.utils.dataset import Dataset
from lookatthisgraph.utils.trainer import Trainer
from lookatthisgraph.nets.SGConv import SGConvNet
from lookatthisgraph.utils.LDataset import LDataset
from lookatthisgraph.utils.LTrainer import LTrainer
from lookatthisgraph.nets.ChebConv2 import ChebConvNet
import datetime as dt



FileLocation="Data/140000"
k_max=2    #=k von K-fold validation
k_size=int(2e3)  #=size of K-fold sample (aka Test+Train Split)

train_config = {
        'learning_rate': 7e-4,
        'scheduling_step_size': 30,        
        'scheduling_gamma': .7,
        'training_target': 'energy',
        'train_split': 2e4, #unnecessary
        'test_split': 2e3,  #unnecessary
        'batch_size': 512,
        'max_epochs': 40,
        'kFold_max' : k_max,
        'kFold_size' : k_size,
        'net': ChebConvNet
    }
              

train_config('net')()


resultlist=[]




for k_crnt in range(k_max):
    train_config['kFold_crnt']=k_crnt
    
    train_set = LDataset([FileLocation], train_config)
    train_config['dataset']=train_set
  
    trainer = LTrainer(train_config)
    trainer.train()
    trainer.load_best_model()
    prediction, truth = trainer.evaluate_test_samples()
    
    prediction=torch.from_numpy(prediction)
    truth=torch.from_numpy(truth[train_config['training_target']].flatten())
    if train_config['training_target']=='energy':
        avrg=torch.mean(torch.div(torch.sub(prediction, truth), truth)).item()
    else:
        avrg=torch.mean(torch.square(torch.sub(prediction, truth))).item()  #Average
    result=torch.tensor([avrg])
    resultlist.append(result)
    
    endresult=torch.cat(resultlist, 0)  

endresult=torch.mean(endresult).item()
STD=torch.std(endresult).item()
print('k-Fold final Accuracy:', endresult)
filename="Acc_"+train_config['net'](1,1).__class__.__name__+"_"+train_config['training_target']+"_"+dt.datetime.now().strftime("%d-%m-%Y_%H-%M")+".txt"
file=open(filename, "w")
file.writelines(['k-Fold final Accuracy: '+str(endresult)+"\n", 'k-Fold standart deviation: '+str(STD)+"\n", "k_max="+str(k_max)+"\n", "k_size="+str(k_size)])
file.close()
