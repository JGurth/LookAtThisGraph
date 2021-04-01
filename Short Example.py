import numpy as np
import torch
from lookatthisgraph.utils.dataset import Dataset
from lookatthisgraph.utils.trainer import Trainer
from lookatthisgraph.utils.LDataset import LDataset
from lookatthisgraph.utils.LTrainer import LTrainer
from lookatthisgraph.nets.PointConv import PointNet
import datetime as dt


FileLocation="Data/140000"


train_config = {
        'learning_rate': 7e-4,
        'scheduling_step_size': 30,        
        'scheduling_gamma': .7,
        'training_target': 'energy',
        'train_split': 1e3,
        'test_split': 1e5,
        'batch_size': 512,
        'max_epochs': 40,
        'net': PointNet
    }
#LDataset hängt von Config ab und muss deswegen in dieser Reihenfolge definiert werden:
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
print('Accuracy:', avrg)
filename="Short_Acc_"+train_config['net'](1,1).__class__.__name__+"_"+train_config['training_target']+"_"+dt.datetime.now().strftime("%d-%m-%Y_%H-%M")+".txt"
file=open(filename, "w")
file.writelines(['Short Accuracy: '+str(avrg)+"\n"])
file.close()