#!/usr/bin/env python

import sys

import numpy as np

from sklearn.metrics import roc_curve, auc


data = np.loadtxt(sys.argv[1])

fpr, tpr, thresholds = roc_curve(data[:,1], data[:, 0])

roc_area = auc(fpr, tpr)

print roc_area
