import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (9, 7)
mpl.rcParams['font.size'] = 15
import numpy as np
import torch
from lookatthisgraph.utils.dataset import Dataset
from lookatthisgraph.utils.trainer import Trainer
import datetime as dt
from lookatthisgraph.utils.LDataset import LDataset
from lookatthisgraph.utils.LTrainer import LTrainer
from lookatthisgraph.nets.EnsembleNet import EnsembleNet
from lookatthisgraph.nets.ConvNet import ConvNet

FileLocation="Data/140000"

SaveNet=False


train_config = {
        'learning_rate': 7e-4,
        'scheduling_step_size': 30,        
        'scheduling_gamma': .7,
        'training_target': 'energy',
        'train_split':5e4,
        'test_split': 1e5,
        'batch_size': 1024,
        'max_epochs': 60,
        'net' : EnsembleNet
    }
#LDataset hängt von Config ab und muss deswegen in dieser Reihenfolge definiert werden:
#train_set = LDataset([FileLocation], train_config)
train_set = Dataset([FileLocation])
train_config['dataset']=train_set 



width=128
conv_depth=3
point_depth=3
lin_depth=5
resultlist=[]
#for width in range(64, 1025, 128):
    #for conv_depth in range(2, 15, 2):
for width in [128, 256]: #range(128,513,128):
    
#TRY/EXCEPT

    train_config['dim']=[width, conv_depth, point_depth, lin_depth]
    #trainer = LTrainer(train_config)
    trainer = Trainer(train_config)
    trainer.train()
    trainer.load_best_model()
    start=dt.datetime.utcnow()
    prediction, truth = trainer.evaluate_test_samples()
    
    end=dt.datetime.utcnow()
    time=dt.timedelta.total_seconds(end-start)/train_config['test_split']
    
    prediction=torch.from_numpy(prediction)
    truth=torch.from_numpy(truth[train_config['training_target']].flatten())
    if train_config['training_target']=='energy':
        avrg=torch.mean(torch.abs(torch.div(torch.sub(prediction, truth), truth))).item()
    else:
        avrg=torch.mean(torch.square(torch.sub(torch.reshape(prediction, (-1,)), truth))).item()  #Average
       
    if SaveNet:
        trainer.save_network_info("SavedNets/TestNet_"+train_config['net'](1,1).__class__.__name__+"_"+train_config['training_target']+"_"+str(avrg)+".p")

    
    print('Accuracy:', str(avrg))
    filename="Results/TestAcc_"+train_config['net'](1,1).__class__.__name__+"_"+str(width)+"_"+str(conv_depth)+"_"+str(point_depth)+"_"+str(lin_depth)+"_"+train_config['training_target']+"_"+dt.datetime.now().strftime("%d-%m-%Y_%H-%M")+".txt"
    file=open(filename, "w")
    file.writelines(['Accuracy: '+str(avrg)+"\n", 'Width='+str(width)+"\n", 'Conv_Depth='+str(conv_depth)+"\n", "Point_Depth="+str(point_depth)+"\n", 'Linear_Depth='+str(lin_depth)+"\n", "Training_Size="+str(train_config["train_split"])+"\n", "Epochs="+str(train_config['max_epochs'])+"\n", "Batch_Size="+str(train_config['batch_size'])+"\n", "Time="+str(time)])
    file.close()
    









































