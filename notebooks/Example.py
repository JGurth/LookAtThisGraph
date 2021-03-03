import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (9, 7)
mpl.rcParams['font.size'] = 15
import numpy as np
import pickle
import torch
from lookatthisgraph.utils.dataset import Dataset
from lookatthisgraph.utils.trainer import Trainer
from lookatthisgraph.utils.model import Model
from lookatthisgraph.utils.LDataset import LDataset
from lookatthisgraph.utils.LTrainer import LTrainer
from lookatthisgraph.nets.EnsembleNet import EnsembleNet

FileLocation="Data/140000"


train_config = {
        'learning_rate': 7e-4,
        'scheduling_step_size': 30,        
        'scheduling_gamma': .7,
        'training_target': 'energy',
        'train_split': 1e4,
        'test_split': 1e5,
        'batch_size': 512,
        'max_epochs': 40,
        'Net': EnsembleNet
    }
#LDataset hängt von Config ab und muss deswegen in dieser Reihenfolge definiert werden:
train_set = LDataset([FileLocation], train_config)
train_config['dataset']=train_set

trainer = LTrainer(train_config)
trainer.train()

plt.figure()
plt.plot(np.arange(len(trainer.train_losses)), trainer.train_losses, label='Training loss')
plt.plot(np.arange(len(trainer.train_losses)), trainer.validation_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.show()
    
trainer.load_best_model()
prediction, truth = trainer.evaluate_test_samples()
prediction=torch.from_numpy(prediction)
truth=torch.from_numpy(truth[train_config['training_target']].flatten())
avrg=torch.mean(torch.square(torch.sub(prediction, truth))).item()  #Average
print(avrg)

plt.figure()
bins = np.linspace(0,3,100)
plt.hist2d(truth['energy'].flatten(), prediction, bins=[bins,bins])
plt.plot(bins, bins)
plt.colorbar()
plt.ylabel(r'$\mathregular{log_{10}(E_{predicted})}$')
plt.xlabel(r'$\mathregular{log_{10}(E_{true})}$')
plt.show()

trainer.save_network_info('test.p')
info = pickle.load(open('test.p', 'rb'))
m = Model(info)
test_set = Dataset([FileLocation], info['normalization_parameters'][0])
p, t = m.evaluate_dataset(test_set, 32)

plt.figure()
bins = np.linspace(0,3,80)
plt.hist2d(t['energy'].flatten(), p.flatten(), bins=[bins, bins])
plt.plot(bins,bins)
plt.ylabel(r'$\ mathregular{log_{10}(E_{predicted})}$')
plt.xlabel(r'$\mathregular{log_{10}(E_{true})}$')
plt.colorbar()
plt.show()




























