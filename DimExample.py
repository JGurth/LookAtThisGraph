import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (9, 7)
mpl.rcParams['font.size'] = 15
import numpy as np
import torch
from lookatthisgraph.utils.dataset import Dataset
from lookatthisgraph.utils.trainer import Trainer
from datetime import datetime, timedelta
from lookatthisgraph.utils.LDataset import LDataset
from lookatthisgraph.utils.LTrainer import LTrainer

FileLocation="Data/140000"
train_config = {
        'learning_rate': 7e-4,
        'scheduling_step_size': 30,        
        'scheduling_gamma': .7,
        'training_target': 'energy',
        'train_split': 2e3,
        'test_split': 1e5,
        'batch_size': 64,
        'max_epochs': 100,
    }
#LDataset hängt von Config ab und muss deswegen in dieser Reihenfolge definiert werden:
train_set = LDataset([FileLocation], train_config)
train_config['dataset']=train_set 



width=254
conv_depth=6
lin_depth=5
resultlist=[]
#for width in range(64, 1025, 128):
    #for conv_depth in range(2, 15, 2):
for lin_depth in range(10,40,10):
    
#TRY/EXCEPT

    train_config['dim']=[width, conv_depth, lin_depth]
    
    
    trainer = LTrainer(train_config)
    trainer.train()
    trainer.load_best_model()
    start=datetime.utcnow()
    prediction, truth = trainer.evaluate_test_samples()
    
    end=datetime.utcnow()
    time=timedelta.total_seconds(end-start)/train_config['test_split']
    prediction=torch.from_numpy(prediction)
    truth=torch.from_numpy(truth[train_config['training_target']].flatten())
    if train_config['training_target']=='energy':
        avrg=torch.mean(torch.div(torch.sub(prediction, truth), truth)).item()
    else:
        avrg=torch.mean(torch.square(torch.sub(prediction, truth))).item()  #Average
    result=torch.tensor([avrg, time, width, conv_depth, lin_depth])
    resultlist.append(result)
    
    endresult=torch.cat(resultlist, 0)  
    #Jede Zeile enthält eine Dimension mit (Accuracy, TrainTime [s], Width, Convolutional Depth, Linear Depth)
endresult.view(-1, 5)
print(endresult)

#plt.figure()
#plt.plot(np.arange(len(trainer.train_losses)), trainer.train_losses, label='Training loss')
#plt.plot(np.arange(len(trainer.train_losses)), trainer.validation_losses, label='Validation loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.yscale('log')
#plt.legend()
#plt.show()
    































