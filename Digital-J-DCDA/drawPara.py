from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
#
sns.set_style('whitegrid')
sns.set_context('talk', font_scale=0.8, rc={'lines.linewidth': 1.2})
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.xlabel('$\lambda_d$')
plt.ylabel('Average accuracy (%)')
x1 = [1, 2, 3, 4, 5, 6, 7, 8]
y1 = [83.5, 84.0, 85.6, 86.8, 86.0, 85.5, 81, 79]
y2 = [82.9, 82.9, 82.9, 82.9, 82.9, 82.9, 82.9, 82.9]
plt.legend(fontsize='x-small',loc='best')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], [0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 1, 5])
plt.twiny()
plt.xlabel('$\lambda_{ca}$')
x2 = [1, 2, 3, 4, 5, 6, 7, 8]
y3 = [82.9, 84.5, 86.6, 86.8, 85.8, 85.0, 83, 81]
plt.plot(x1, y1,color='r', marker='^',label='J-DCDA-d')
# y2 = [82.9, 82.9, 82.9, 82.9, 82.9, 82.9, 82.9, 82.9]
plt.plot(x1, y2, color='g',linestyle='--',linewidth=2,label='baseline')
plt.plot(x2, y3,color='b', marker='o',label='J-DCDA-ca')
# plt.plot(x1, y2, color='g',linestyle='--',linewidth=2,label='baseline')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['1e-4', '5e-4', '8e-4', 0.001, 0.05, 0.1, 1, 5])
plt.ylim((75.0, 90.0))
plt.legend(fontsize='x-small',loc='best')


plt.subplot(1,2,2)
# plt.xlabel(r'Number of Iterations ($\times10^{4}$)')
plt.xlabel('$K-threshold$')
plt.ylabel('Average accuracy (%)')
x1 = [1, 2, 3, 4, 5, 6]
y1 = [85.0,85.3,85.6,86.2,86.8,84.8]
y2 = [85.0,85.0,85.0,85.0,85.0,85.0]
plt.plot(x1, y1,color='r', marker='^',label='J-DCDA')
plt.plot(x1, y2, color='g',linestyle='--',linewidth=2,label='baseline')
plt.xticks([1, 2, 3, 4, 5, 6], [0.4,0.5, 0.6, 0.7, 0.8, 0.9])
plt.ylim((80.0, 90.0))

plt.legend(fontsize='x-small')
# plt.legend(bbox_to_anchor=[0.3, 1])
plt.legend(fontsize='x-small')
plt.savefig("parameter.eps")

