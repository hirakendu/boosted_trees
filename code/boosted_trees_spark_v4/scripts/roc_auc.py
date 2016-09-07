#!/usr/bin/env python

import sys

import numpy as np

from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


data = np.loadtxt(sys.argv[1])

fpr, tpr, thresholds = roc_curve(data[:,1], data[:, 0])

roc_area = auc(fpr, tpr)

print roc_area

file = open(sys.argv[2], 'w')
for i in np.arange(0, thresholds.size - 1):
  file.write(str.format("{0}\t{1}\t{2}\n", fpr[i], tpr[i], thresholds[i]))
file.close()

fig, ax = plt.subplots()

#plt.plot(data[:,0], data[:,1], 'ko', data[:,0], data[:,1], 'k-')
# plt.plot(data[:,0], data[:,1], 'k-')
plt.plot(fpr, tpr, 'k-')

# xticks = list({0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1})
xticks = list({0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0})
yticks = list({0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0})
plt.xticks(xticks)
plt.yticks(yticks)
#xtickLabels = ax.get_xticklabels()
#plt.setp(xtickLabels, rotation=90, fontsize=8)
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.axis([0,1,0,1])
fig.tight_layout()
plt.grid(True)
plt.savefig(sys.argv[3], format='pdf')
