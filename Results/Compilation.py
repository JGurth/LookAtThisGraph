import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl



#Batchsize (CEnsemble1, 100k, 40 Epochs); 
Batchsize=[256, 512, 1024, 1536, 2048, 2560] #Batchsize
BatchAcc=[0.252, 0.234, 0.232, 0.239, 0.251, 0.253] #Acc
BatchSTD=[0.021, 0.003, 0.004, 0.015, 0.022, 0.044]



#Ensemble vs pure Pointnet vs Basic CEnsemble vs CEnsembleRelu (100k, 60 Epochs, 512 Batch)
Nets=["Basic Ensemble", 'PointNet Pure', 'CEnsemble', 'CEnsemble w. ReLU']
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

CEnConvhops=[1, 2, 3]
CEnConvhopsAcc=[0.242, 0.236, 0.241] #acc
CEnConvhopsSTD=[0.012, 0.006, 0.018] #STD


#CEnsemble ConvDepth (CEnsemble1, 100k, 40 Epochs, 512 Batch) 

CEnConv=[1, 2, 3, 4, 5, 6]
CEnConvAcc=[0.235, 0.237, 0.235, 0.232, 0.248, 0.336] #acc
CEnConvSTD=[0.003, 0.006, 0.006, 0.004, 0.027, 0.205] #STD

#Conv alternative (CEnsemble1, 100k, 40 Epochs, 512 Batch)

Conv2=[1, 2, 3, 4, 5, 6, 7]
Conv2Acc=[0.261, 0.232, 0.228, 0.234, 0.227, 0.229, 0.229]
Conv2STD=[0.026, 0.005, 0.005, 0.006, 0.004, 0.005, 0.007]



#CEnsemble LinDepth (CEnsemble1, 100k, 40 Epochs, 512 Batch) 

CEnLin=[1, 2, 3, 4, 5, 6, 7, 9] 
CEnLinAcc=[0.236, 0.235, 0.241, 0.232, 0.234, 0.231, 0.235, 0.402]
CEnLinSTD=[0.006, 0.005, 0.005, 0.003, 0.003, 0.004, 0.007, 0.255]

#Lin alternative (CEnsemble1, 100k, 40 Epochs, 512 Batch)

Lin2=[2, 3, 4, 5, 6, 7, 8, 9]
Lin2Acc=[0.233, 0.277, 0.237, 0.228, 0.229, 0.227, 0.226, 0.230]
Lin2STD=[0.004, 0.042, 0.004, 0.005, 0.004, 0.003, 0.004, 0.006]


#CEnsemble PointDepth (CEnsemble1, 100k, 40 Epochs, 512 Batch) 

CEnPoint=[1, 2, 3, 4, 5, 6] 
CEnPointAcc=[0.229, 0.234, 0.232, 0.233, 0.268, 0.262] #acc
CEnPointSTD=[0.003, 0.007, 0.005, 0.005, 0.060, 0.056] #STD

#Point alternative (CEnsemble1, 100k, 40 Epochs, 512 Batch)
Point2=[1, 2, 3, 4, 5, 6]
Point2Acc=[0.2276, 0.2285, 0.2319, 0.2305, 0.2323, 0.2380]
Point2STD=[0.0021, 0.0049, 0.0062, 0.0049, 0.0014, 0.0042]


#CEnsemble Width (CEnsemble1, 100k, 40 Epochs, 512 Batch) 

CEnWidth=[512, 256, 128, 64, 32, 16, 12, 8, 4] 
CEnWidthAcc=[0.343, 0.232, 0.232, 0.233, 0.230, 0.238, 0.246, 0.241, 0.260]
CEnWidthSTD=[0.020, 0.004, 0.005, 0.002, 0.002, 0.015, 0.028, 0.009, 0.018]


#Width alternative (CEnsemble1, 100k, 40 Epochs, 512 Batch)
#!!! 256, 8 !!! 
Width2=[128, 64, 32, 16]
Width2Acc=[0.234, 0.227, 0.229, 0.230]
Width2STD=[0.005, 0.006, 0.007, 0.005]


#Width,  Time (CEnsemble1, 100k, 40 Epochs, 512 Batch)
Widthtime=[0.682, 0.426, 0.3, 0.232, 0.186, 0.173, 0.174, 0.171] #ms


x=Point2
y=Point2Acc
z=Point2STD



plt.figure()
#plt.plot(x, y, "bo", markersize=3)
plt.errorbar(x, y, z, fmt="ro", elinewidth=0.5, markersize=3)

plt.title("Point-Depth with other secondary parameters")
plt.xlabel('Point-Depth')
plt.ylabel(r'Accuracy ($\Delta $Log$_{10} (E))$')



#plt.xscale("log")
plt.xticks(x)
#plt.yscale('log')
#plt.yticks(np.arange(0.22, 0.3, 0.02))
#plt.ylim(0.225, 0.24)


plt.grid(axis="y")
#plt.legend()

plt.tight_layout()
plt.savefig("Plots/Point2.pdf")
plt.savefig("Plots/Point2.png")

plt.show()



























