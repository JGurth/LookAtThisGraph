import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (9, 7)
mpl.rcParams['font.size'] = 15
import numpy as np
import pickle
from lookatthisgraph.utils.dataset import Dataset
from lookatthisgraph.utils.model import Model
import datetime as dt


def LocalAcc(Net, DataLocation="Data/140000", nblocks=20):

    info = pickle.load(open(Net, 'rb'))
    m = Model(info)
    test_set = Dataset([DataLocation], info['normalization_parameters'][0])
    prediction, truth = m.evaluate_dataset(test_set, 32)
    
    truth=truth['energy'].flatten()    
    acc=np.absolute(prediction-truth)


    erange=np.amax(truth)-np.amin(truth) 
    lblock=erange/nblocks #number of energy-intervals for which the error is calculated

    
    plot_energy=[]
    plot_acc=[]
    
    for j in range(nblocks):
        block1=[]
        block2=[]
        
        for i in range(len(truth)):    #if element fits that energy interval    
            if truth[i] < (j+1)*lblock and truth[i] > j*lblock:
                block1.append(acc[i])
                
        std=np.std(block1)      #reduction to points within one standard deviation
        for i in block1:
            if i<std:
                block2.append(i)
        
        plot_energy.append((0.5+j)*lblock)
        plot_acc.append(np.mean(block2))
    
                
    plot_energy=np.power(10, plot_energy)            
    #achsenbeschriftung
    plt.figure()
    plt.plot(plot_energy, plot_acc)
    plt.xlabel('Energy [Log(GeV)]')
    plt.xscale("log")
    plt.ylabel('Accuracy')
    plt.savefig("Plots/AccPlot_"+dt.datetime.now().strftime("%d-%m-%Y_%H-%M"), format='pdf')
    plt.savefig("Plots/AccPlot_"+dt.datetime.now().strftime("%d-%m-%Y_%H-%M"), format='png')

