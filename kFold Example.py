import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (9, 7)
mpl.rcParams['font.size'] = 15
import numpy as np
import torch
from lookatthisgraph.utils.dataset import Dataset
from lookatthisgraph.utils.trainer import Trainer
from lookatthisgraph.nets.ConvNet import ConvNet
from lookatthisgraph.nets.ChebConv2 import ChebConvNet
from lookatthisgraph.nets.CEnsembleNet1 import CEnsembleNet1
from lookatthisgraph.nets.EnsembleNet2 import EnsembleNet2
from lookatthisgraph.nets.EnsembleNet import EnsembleNet
from lookatthisgraph.nets.CEnsembleNet1Relu import CEnsembleNet1Relu
from lookatthisgraph.nets.CEnsembleNet import CEnsembleNet
from lookatthisgraph.nets.PointConv import PointNet
import datetime as dt



FileLocation="Data/140000"
k_max=10    #=k von K-fold validation
k_size=int(2e5)  #=size of K-fold sample (aka Test+Train Split)
train_set = Dataset([FileLocation])
SaveNet=True
SavePlot=True

train_config = {
        'learning_rate': 7e-4,
        'scheduling_step_size': 30,        
        'scheduling_gamma': .7,
        'training_target': 'energy',
        'train_split': 2e4, #unnecessary/ignored
        'test_split': 2e3,  #unnecessary/ignored
        'batch_size': 1024,
        'max_epochs': 120,
        'kFold_max' : k_max,
        'kFold_size' : k_size,
        'net': EnsembleNet2,
        'dataset': train_set
    }
              




resultlist=[]




for k_crnt in range(k_max):
	train_config['kFold_crnt']=k_crnt
  
	trainer = Trainer(train_config)
	trainer.train()
	trainer.load_best_model()
	start=dt.datetime.utcnow()
	prediction, truth = trainer.evaluate_test_samples()
    
	end=dt.datetime.utcnow()
	time=dt.timedelta.total_seconds(end-start)/(k_size/k_max) 
    
	prediction=torch.from_numpy(prediction)
	truth=torch.from_numpy(truth[train_config['training_target']].flatten())
	if train_config['training_target']=='energy':
		avrg=torch.mean(torch.abs(torch.sub(prediction, truth))).item()
	else:
		avrg=torch.mean(torch.square(torch.sub(torch.reshape(prediction, (-1,)), truth))).item()  #Average
	result=torch.tensor([avrg])
	resultlist.append(result)
    
	endresult0=torch.cat(resultlist, 0)  
    
	if SaveNet:
        	trainer.save_network_info("Results/Ensemble3/Net_"+"N"+str(k_crnt)+"_"+train_config['net'](1,1).__class__.__name__+"_"+train_config['training_target']+"_"+".p")


	if SavePlot:
		plt.figure()
		plt.plot(np.arange(len(trainer.train_losses)), trainer.train_losses, label='Training loss')
		plt.plot(np.arange(len(trainer.train_losses)), trainer.validation_losses, label='Validation loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.yscale('log')
		plt.legend()
		plt.savefig("Results/Ensemble3/Plot_"+"N"+str(k_crnt)+"_"+train_config['net'](1,1).__class__.__name__+"_"+train_config['training_target']+"_"+str(avrg)+".png")

    

endresult=torch.mean(endresult0).item()
STD=torch.std(endresult0).item()
print('k-Fold final Accuracy:', endresult)
filename="Results/Ensemble3/Acc_"+train_config['net'](1,1).__class__.__name__+"_"+train_config['training_target']+"_"+dt.datetime.now().strftime("%d-%m-%Y_%H-%M")+".txt"
file=open(filename, "w")
file.writelines(['k-Fold final Accuracy: '+str(endresult)+"\n", 'k-Fold standart deviation: '+str(STD)+"\n", "Values: "+str(endresult0)+"\n", "k_max="+str(k_max)+"\n", "k_size="+str(k_size)+"\n", "Epochs="+str(train_config['max_epochs'])+"\n", "Batch_Size="+str(train_config['batch_size'])+"\n", "Time="+str(time)])
file.close()





