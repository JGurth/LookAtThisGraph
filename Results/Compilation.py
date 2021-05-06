import matplotlib.pyplot as plt



#Batchsize (CEnsemble1, 100k, 60 Epochs); 
# !!! Evtl kFold !!!
Batchsize=[256, 512, 1024, 2048, 2560, 3072] #Batchsize
BatchAcc=[0.335, 0.24, 0.234, 0.231, 0.278, 0.246] #Acc



#Ensemble vs pure Pointnet vs Basic CEnsemble (100k, 60 Epochs, 512 Batch)
NetsAcc=[0.252, 0.254, 0.246] #Acc
NatsSTD=[0.004, 0.003, 0.011] #STD



#Ensemble Point-Hops (100k, 60 Epochs, 512 Batch)
Pointhops=[1, 2, 3] #Hops
PointhopsAcc=[0.236, 0.252, 0.248] #Acc
PointhopsSTD=[0.007, 0.004, 0.014] #STD

#Ensemble ConvDepth (100k, 60 Epochs, 1024 Batch)
EnsembleConv=[2, 3, 4, 5, 6]
EnsembleConvAcc=[0.262, 0.250, 0.254, 0.252, 0.259]

#Ensemble LinDepth (100k, 60 Epochs, 1024 Batch)
EnsembleLin=[9, 7, 5, 3]
EnsembleLinAcc=[0.247, 0.252, 0.250, 0.301]

#Ensemble PointDepth (100k, 60 Epochs, 1024 Batch)
EnsemblePoint=[6, 5, 4, 3, 2, 1]
EnsemblePointAcc=[0.251, 0.257, 0.254, 0.250, 0.261, 0.233]

#Ensemble Width (50k, 60 Epochs, 1024 Batch)

EnsembleWidth=[320, 256, 192, 64, 32, 16]
EnsembleWidthAcc=[0.297, 0.258, 0.272, 0.256, 0.24, 0.24, 0.236]



#CEnsemble Aggr (CEnsemble1, 100k, 60 Epochs, 512 Batch)

CEnAggr=["max", "add", "mean"]
CEnAggrAcc=[0.236, 0.239, 0.238] #acc
CEnAggrSTD=[0.006, 0.007, 0.008] #STD


#CEnsemble ConvHops (CEnsemble1, 100k, 60 Epochs, 512 Batch)

CEnConv=[1, 2, 3]
CEnConvAcc=[0.242, 0.236, 0.241] #acc
CEnConvSTD=[0.012, 0.006, 0.018] #STD


#CEnsemble ConvDepth (CEnsemble1, 100k, 40 Epochs, 512 Batch) 

CEnConv=[1, 2, 3, 4, 6]
CEnConvAcc=[0.235, 0.235, 0.237, 0.246, 0.336] #acc
CEnConvSTD=[0.006, 0.005, 0.008, 0.024, 0.2] #STD


#CEnsemble LinDepth (CEnsemble1, 100k, 40 Epochs, 512 Batch) 

CEnLin=[1, 2, 3, 4, 6, 7, 8, 9]
CEnLinAcc=[0.231, 0.235, 0.241, 0.232, 0.231, 0.229, 0.231, 0.231]
CEnLinSTD=[0.003, 0.005, 0.005, 0.003, 0.004, 0.002, 0.004, 0.004]


#CEnsemble PointDepth (CEnsemble1, 100k, 40 Epochs, 512 Batch) 

CEnPoint=[1, 2, 3, 4, 6]
CEnPointAcc=[0.235, 0.233, 0.231, 0.232, 0.262] #acc
CEnPointSTD=[0.005, 0.006, 0.005, 0.004, 0.056] #STD


#CEnsemble Width (CEnsemble1, 100k, 40 Epochs, 512 Batch) !!
CEnWidth=[256, 128, 64, 32, 16]
CEnWidthAcc=[0.232, 0.232, 0.233, 0.230, 0.238]
CEnWidthSTD=[0.004, 0.005, 0.002, 0.002, 0.015]


x=CEnWidth
y=CEnWidthAcc


#for i in range(len(y)):
 #   y[i]=10**y[i]







plt.figure()
plt.plot(x, y, "ro", markersize=3)
plt.xlabel('')
plt.ylabel('Log(E')
#plt.yscale('log')
plt.legend()
plt.show()



























