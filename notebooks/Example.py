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


FileLocation="Data/140000"
train_set = Dataset([FileLocation])

train_config = {
        'learning_rate': 7e-4,
        'scheduling_step_size': 30,        
        'scheduling_gamma': .7,
        'training_target': 'energy',
        'dataset': train_set,
        'train_split': 2e3,
        'test_split': 1e5,
        'batch_size': 1024,
        'max_epochs': 100,
    }

trainer = Trainer(train_config)
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

plt.figure()
bins = np.linspace(0,3,100)
plt.hist2d(truth, prediction, bins=[bins,bins])
plt.plot(bins, bins)
plt.colorbar()
plt.ylabel(r'$\\mathregular{log_{10}(E_{predicted})}$')
plt.xlabel(r'$\\mathregular{log_{10}(E_{true})}$')
plt.show()

trainer.save_network_info('test.p')
info = pickle.load(open('test.p', 'rb'))
m = Model(info)
test_set = Dataset([FileLocation], info['normalization_parameters'][0])
m.set_device_type('cpu')
p, t = m.evaluate_dataset(test_set, 1024)

plt.figure()
bins = np.linspace(0,3,80),
plt.hist2d(t.flatten(), p.flatten(), bins=[bins, bins])
plt.plot(bins,bins)
plt.ylabel(r'$\\mathregular{log_{10}(E_{predicted})}$')
plt.xlabel(r'$\\mathregular{log_{10}(E_{true})}$')
plt.colorbar()
plt.show()




























