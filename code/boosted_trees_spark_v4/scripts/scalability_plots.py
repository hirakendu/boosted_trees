#!/usr/bin/env python

import sys

import numpy

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


data = numpy.loadtxt(sys.argv[1])

fig, ax = plt.subplots()

plt.rc('axes', color_cycle = ['r', 'g', 'b', 'c', 'm', 'y', 'k'])
for i in range(1, data[0,:].size):
  plt.plot(data[:,0], data[:,i], '-o', label='Depth {0}'.format(i))
#  plt.plot(data[:,0], data[:,i])
# plt.loglog(data[:,1], data[:,1], 'k-')


# plt.title('Scatter plot')
plt.xlabel('Workers')
plt.ylabel(sys.argv[3])
xticks = data[:,0]
plt.xticks(xticks)
plt.legend(loc='best')  # Others: 'upper-left', 'upper-right'.
gap_x= 0.1 * (max(data[:,0])-min(data[:,0]) + 0.1)
gap_y= 0.1 * (max(data[:,1])-min(data[:,1]) + 0.1)
plt.axis([min(data[:,0]) - gap_x, max(data[:,0]) + gap_x,
 min(map(min, data[:,1:(data[0,:].size)])) - gap_y, max(map(max, data[:,1:(data[0,:].size)])) + gap_y])
fig.tight_layout()
fig.set_size_inches(8, 8)
plt.grid(True)
plt.savefig(sys.argv[2], format='pdf')
