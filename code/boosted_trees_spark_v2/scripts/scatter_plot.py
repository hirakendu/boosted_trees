#!/usr/bin/env python

import sys

import numpy

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


data = numpy.loadtxt(sys.argv[1])

fig, ax = plt.subplots()

#plt.plot(data[:,0], data[:,1], 'ko', data[:,0], data[:,1], 'k-')
plt.plot(data[:,1], data[:,0], 'ko')
plt.plot(data[:,1], data[:,1], 'b-', alpha=0.5)
# plt.loglog(data[:,1], data[:,1], 'k-')

# plt.title('Scatter plot')
plt.xlabel('Actual')
plt.ylabel('Predicted')
gap= 0.1 * (max(data[:,1])-min(data[:,1]) + 0.1)
plt.axis([min(data[:,1]) - gap, max(data[:,1]) + gap,
 min(data[:,0]) - gap, max(data[:,1]) + gap])
fig.tight_layout()
fig.set_size_inches(8, 8)
plt.grid(True)
plt.savefig(sys.argv[2], format='pdf')
