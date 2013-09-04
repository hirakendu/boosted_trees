#!/usr/bin/env python

import sys

import numpy

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


data = numpy.loadtxt(sys.argv[1])

#plt.plot(data[:,0], data[:,1], 'ko', data[:,0], data[:,1], 'k-')
plt.plot(data[:,0], data[:,1], 'k-')
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.axis([0,1,0,1])
plt.grid(True)
plt.savefig(sys.argv[2], format='pdf')
