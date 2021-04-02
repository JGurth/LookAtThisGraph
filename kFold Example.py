import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (9, 7)
mpl.rcParams['font.size'] = 15
import numpy as np
import torch
from lookatthisgraph.utils.dataset import Dataset
from lookatthisgraph.utils.trainer import Trainer
from lookatthisgraph.nets.ConvNet import ConvNet
from lookatthisgraph.nets.EnsembleNet import EnsembleNet
import datetime as dt



FileLocation="Data/140000"
k_max=10    #=k von K-fold validation
k_size=int(1e5)  #=size of K-fold sample (aka Test+Train Split)
train_set = Dataset([FileLocation])
SaveNet=True

train_config = {
        'learning_rate': 7e-4,
        'scheduling_step_size': 30,        
        'scheduling_gamma': .7,
        'training_target': 'energy',
        'train_split': 2e4, #unnecessary/ignored
        'test_split': 2e3,  #unnecessary/ignored
        'batch_size': 512,
        'max_epochs': 60,
        'kFold_max' : k_max,
        'kFold_size' : k_size,
        'net': EnsembleNet,
        'dataset': train_set
    }
              




resultlist=[]




for k_crnt in range(k_max):
    train_config['kFold_crnt']=k_crnt
  
    trainer = Trainer(train_config)
    trainer.train()
    trainer.load_best_model()
    prediction, truth = trainer.evaluate_test_samples()
    
    prediction=torch.from_numpy(prediction)
    truth=torch.from_numpy(truth[train_config['training_target']].flatten())
    if train_config['training_target']=='energy':
        avrg=torch.mean(torch.div(torch.sub(prediction, truth), truth)).item()
    else:
        avrg=torch.mean(torch.square(torch.sub(torch.reshape(prediction, (-1,)), truth))).item()  #Average
    result=torch.tensor([avrg])
    resultlist.append(result)
    
    endresult0=torch.cat(resultlist, 0)  
    
    if SaveNet:
        trainer.save_network_info("Results/Net_"+train_config['net'](1,1).__class__.__name__+"_"+train_config['training_target']+"_"+str(avrg)+".p")

    

endresult=torch.mean(endresult0).item()
STD=torch.std(endresult0).item()
print('k-Fold final Accuracy:', endresult)
filename="Results/Acc_"+train_config['net'](1,1).__class__.__name__+"_"+train_config['training_target']+"_"+dt.datetime.now().strftime("%d-%m-%Y_%H-%M")+".txt"
file=open(filename, "w")
file.writelines(['k-Fold final Accuracy: '+str(endresult)+"\n", 'k-Fold standart deviation: '+str(STD)+"\n", "k_max="+str(k_max)+"\n", "k_size="+str(k_size)])
file.close()





