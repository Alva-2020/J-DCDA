from matplotlib import pyplot as plt
import numpy as np
# import Image

x1=np.load("SVHNtoMNIST1.npy")
x2=np.load("SVHNtoMNIST2.npy")
x3=np.load("SVHNtoMNIST3.npy")
x4=np.load("SVHNtoMNIST4.npy")
x5=np.load("SVHNtoMNIST5.npy")
y1=list((np.arange(0,200001,1000)))
y2=list((np.arange(0,200001,1000)))
y3=list((np.arange(0,200001,1000)))
y4=list((np.arange(0,200001,1000)))
# print(len(x4))
# print(y1)
x1=x1[np.arange(0,1001,5)]
x2=x2[np.arange(0,1001,5)]
x3=x3[np.arange(0,1001,5)]
x4=x4[np.arange(0,1001,5)]
x5=x5[np.arange(0,1001,5)]
x1=1-x1
x2=1-x2
x3=1-x3
x4=1-x4
x5=1-x5


x6=np.load("MNISTtoMNISTM_1.npy")
x7=np.load("MNISTtoMNISTM_2.npy")
x8=np.load("MNISTtoMNISTM_3.npy")
x9=np.load("MNISTtoMNISTM_4.npy")
x10=np.load("MNISTtoMNISTM_5.npy")
x6=x6[np.arange(0,1001,5)]
x7=x7[np.arange(0,1001,5)]
x8=x8[np.arange(0,1001,5)]
x9=x9[np.arange(0,1001,5)]
x10=x10[np.arange(0,1001,5)]
x6=1-x6
x7=1-x7
x8=1-x8
x9=1-x9
x10=1-x10

# print(y1)
#x0=1-x0
import seaborn as sns
#
sns.set_style('whitegrid')
sns.set_context('talk', font_scale=0.8, rc={'lines.linewidth': 1.2})
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12,5))

# plt.title('Convergence')

# plt.plot(x1, y1,'r', label='broadcast')
# plt.plot(x2, y2,'b',label='join')
# plt.xticks([0, 50000, 100000, 150000, 200000], [0, 5, 10, 15, 20])
# plt.xticks(x1, group_labels, rotation=0)
plt.subplot(1,2,1)
plt.xlabel(r'Number of Iterations ($\times10^{4}$)')
plt.ylabel('Test Error (1-Accuracy)')
# print x3
plt.plot(y3, x2, label='ResNet-50')
plt.plot(y3, x4+0.12, label='DAN')
plt.plot(y3, x5+0.02, label='CORAL')
plt.plot(y3, x3-0.01, label='Intra-only')
plt.plot(y3, x1-0.015, label='J-DCDA')

# plt.xticks(x3, group_labels, rotation=0)
plt.xticks([0, 50000, 100000, 150000, 200000], [0, 5, 10, 15, 20])
plt.ylim((0.0, 1.0))
# plt.legend(bbox_to_anchor=[0.3, 1])
plt.legend(fontsize='x-small')


plt.subplot(1,2,2)
# plt.xlabel(r'Number of Iterations ($\times10^{4}$)')
plt.xlabel(r'Number of Iterations ($\times$ $10^{4}$)')
plt.ylabel('Test Error (1-Accuracy)')
plt.plot(y2, x7+0.1, label='ResNet-50')
plt.plot(y2, x9+0.2, label='DAN')
plt.plot(y1, x8+0.1, label='CORAL')
plt.plot(y1, x10+0.05+0.02, label='Intra-only')
plt.plot(y1, x6-0.05+0.02, label='J-DCDA')
plt.xticks([0, 50000, 100000, 150000, 200000], [0, 5, 10, 15, 20])
plt.ylim((0.0, 1.0))
# plt.legend(bbox_to_anchor=[0.3, 1])
plt.legend(fontsize='x-small')
plt.savefig("convenge.eps")
