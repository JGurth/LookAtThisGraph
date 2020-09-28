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
width=1024
conv_depth=10
resultlist=[]
#for width in range(64, 1025, 64):
    #for conv_depth in range(2, 11):
for lin_depth in range(15,20):
    


    start=datetime.utcnow()
    train_config = {
            'learning_rate': 7e-4,
            'scheduling_step_size': 30,        
            'scheduling_gamma': .7,
            'training_target': 'energy',
            'dim': [width, conv_depth, lin_depth],
            'dataset': train_set,
            'train_split': 2e3,
            'test_split': 1e5,
            'batch_size': 1024,
            'max_epochs': 100,
        }
    
    trainer = Trainer(train_config)
    trainer.train()
    trainer.load_best_model()
    prediction, truth = trainer.evaluate_test_samples()
    
    end=datetime.utcnow()
    time=timedelta.total_seconds(end-start)
    prediction=torch.from_numpy(prediction)
    truth=torch.from_numpy(truth[train_config['training_target']].flatten())
    avrg=torch.mean(torch.sub(prediction, truth)).item()
    result=torch.tensor([avrg, time, width, conv_depth, lin_depth])
    resultlist.append(result)

endresult=torch.cat(resultlist, 0)
print(endresult)

#plt.figure()
#plt.plot(np.arange(len(trainer.train_losses)), trainer.train_losses, label='Training loss')
#plt.plot(np.arange(len(trainer.train_losses)), trainer.validation_losses, label='Validation loss')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.yscale('log')
#plt.legend()
#plt.show()
    

































