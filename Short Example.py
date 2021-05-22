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
from lookatthisgraph.nets.EnsembleNet2 import EnsembleNet2
from lookatthisgraph.nets.CEnsembleNet1 import CEnsembleNet1
import datetime as dt


FileLocation="Data/140000"
SavePlot=True


for i in [1]:
	train_config = {
	        'learning_rate': 7e-4,
	        'scheduling_step_size': 30,        
	        'scheduling_gamma': .7,
	        'training_target': 'energy',
	        'train_split': 2e5,
	        'test_split': 2e4,
	        'batch_size': 1024,
	        'max_epochs': 200,
	        'net': EnsembleNet2
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

	if SavePlot:
		plt.figure()
		plt.plot(np.arange(len(trainer.train_losses)), trainer.train_losses, label='Training loss')
		plt.plot(np.arange(len(trainer.train_losses)), trainer.validation_losses, label='Validation loss')
		plt.title(train_config['net'](1,1).__class__.__name__+"_"+train_config['training_target'])
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.yscale('log')
		plt.legend()
		plt.savefig("Results/Ensemble/Plot_"+train_config['net'](1,1).__class__.__name__+"_"+train_config['training_target']+"_"+dt.datetime.now().strftime("%d-%m-%Y_%H-%M")+".pdf")





	filename="Results/Ensemble/1Acc_"+train_config['net'](1,1).__class__.__name__+"_"+train_config['training_target']+"_"+dt.datetime.now().strftime("%d-%m-%Y_%H-%M")+".txt"

	file=open(filename, "w")
	file.writelines(['Accuracy: '+str(avrg)+"\n", "Training_Size="+str(train_config["train_split"])+"\n", "Epochs="+str(train_config['max_epochs'])+"\n", "Batch_Size="+str(train_config['batch_size'])+"\n", "Time="+str(time)])
	file.close()



