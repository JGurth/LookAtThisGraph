import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (9, 7)
mpl.rcParams['font.size'] = 15
import numpy as np
import torch
from lookatthisgraph.utils.dataset import Dataset
from lookatthisgraph.utils.trainer import Trainer
from lookatthisgraph.utils.LDataset import LDataset
from lookatthisgraph.utils.LTrainer import LTrainer
from lookatthisgraph.nets.PointConv import PointNet
from lookatthisgraph.nets.EnsembleNet import EnsembleNet
from lookatthisgraph.nets.EnsembleNet1 import EnsembleNet1
from lookatthisgraph.nets.EnsembleNet3 import EnsembleNet3
from lookatthisgraph.nets.CEnsembleNet1 import CEnsembleNet1
import datetime as dt


FileLocation="Data/140000"

for i in [256, 512, 1024, 2048]:
	train_config = {
	        'learning_rate': 7e-4,
	        'scheduling_step_size': 30,        
	        'scheduling_gamma': .7,
	        'training_target': 'energy',
	        'train_split': 1e5,
	        'test_split': 1e5,
	        'batch_size': i,
	        'max_epochs': 60,
	        'net': CEnsembleNet1
	    }
	#LDataset hängt von Config ab und muss deswegen in dieser Reihenfolge definiert werden:
	train_set = LDataset([FileLocation], train_config)
	train_config['dataset']=train_set

	trainer = LTrainer(train_config)
	trainer.train()
	trainer.load_best_model()

	start=dt.datetime.utcnow()
	prediction, truth = trainer.evaluate_test_samples()

	end=dt.datetime.utcnow()
	time=dt.timedelta.total_seconds(end-start)/train_config['test_split']


	prediction=torch.from_numpy(prediction)
	truth=torch.from_numpy(truth[train_config['training_target']].flatten())
	if train_config['training_target']=='energy':
	    avrg=torch.mean(torch.abs(torch.sub(prediction, truth))).item()
	else:
	    avrg=torch.mean(torch.square(torch.sub(torch.reshape(prediction, (-1,)), truth))).item()  #Average
	print('Accuracy:', avrg)

	filename="Results/BatchSize/Acc_"+train_config['net'](1,1).__class__.__name__+"_"+train_config['training_target']+"_"+dt.datetime.now().strftime("%d-%m-%Y_%H-%M")+".txt"

	file=open(filename, "w")
	file.writelines(['Accuracy: '+str(avrg)+"\n", "Training_Size="+str(train_config["train_split"])+"\n", "Epochs="+str(train_config['max_epochs'])+"\n", "Batch_Size="+str(train_config['batch_size'])+"\n", "Time="+str(time)])
	file.close()



