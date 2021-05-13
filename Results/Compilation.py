import matplotlib.pyplot as plt



#Batchsize (CEnsemble1, 100k, 40 Epochs); 
Batchsize=[256, 512, 1024, 1536, 2048, 2560] #Batchsize
BatchAcc=[0.252, 0.234, 0.232, 0.239, 0.251, 0.253] #Acc
BatchSTD=[0.021, 0.003, 0.004, 0.015, 0.022, 0.044]



#Ensemble vs pure Pointnet vs Basic CEnsemble vs CEnsembleRelu (100k, 60 Epochs, 512 Batch)
NetsAcc=[0.252, 0.254, 0.246, 0.243] #Acc
NetsSTD=[0.004, 0.003, 0.011, 0.020] #STD



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
CEnConvAcc=[0.235, 0.237, 0.235, 0.232, 0.248, 0.336] #acc
CEnConvSTD=[0.003, 0.006, 0.006, 0.004, 0.027, 0.205] #STD


#CEnsemble LinDepth (CEnsemble1, 100k, 40 Epochs, 512 Batch) 

CEnLin=[1, 2, 3, 4, 6, 7, 9]
CEnLinAcc=[0.236, 0.235, 0.241, 0.232, 0.231, 0.235, 0.402]
CEnLinSTD=[0.006, 0.005, 0.005, 0.003, 0.004, 0.007, 0.255]


#CEnsemble PointDepth (CEnsemble1, 100k, 40 Epochs, 512 Batch) 

CEnPoint=[1, 2, 3, 4, 5, 6] #1, 5
CEnPointAcc=[0.229, 0.234, 0.232, 0.233, 0.268, 0.262] #acc
CEnPointSTD=[0.003, 0.007, 0.005, 0.005, 0.060, 0.056] #STD


#CEnsemble Width (CEnsemble1, 100k, 40 Epochs, 512 Batch) 
CEnWidth=[256, 128, 64, 32, 16, 12, 8, 4]
CEnWidthAcc=[0.232, 0.232, 0.233, 0.230, 0.238, 0.246, 0.241, 0.260]
CEnWidthSTD=[0.004, 0.005, 0.002, 0.002, 0.015, 0.028, 0.009, 0.018]


x=CEnWidth
y=CEnWidthAcc


#for i in range(len(y)):
 #   y[i]=10**y[i]




plt.style.use("seaborn")


plt.figure()
plt.plot(x, y, "ro", markersize=3)
plt.xlabel('')
plt.ylabel('Log(E')
plt.xscale("log")
#plt.yscale('log')
plt.legend()
plt.show()



























