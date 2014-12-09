__author__ = 'Robert Hogan'
'''
Script to plot predictions of model vs. true value for test set
'''


import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt('prediction_out')
y=data[:,0]
y_predict=data[:,1]
plt.close('all')
plt.figure(2)
H,xedges,yedges=np.histogram2d(y,y_predict, bins=20)
level=np.linspace(0,np.round(2*np.max(np.log(np.transpose(H+1))))/2.0,40)
plt.contourf(xedges[:-1],yedges[:-1],np.log(np.transpose(H+1)),levels=level,cmap='hot',drawedges=False)
plt.contourf(xedges[:-1],yedges[:-1],np.log(np.transpose(H+1)),levels=level,cmap='hot',drawedges=False)

plt.plot([min(y),max(y)],[min(y),max(y)],'-',color='black',alpha=0.9, linewidth=2)
plt.xlim((xedges[0],xedges[-2]))
plt.ylim((min(y),yedges[-2]))

cbar=plt.colorbar()

plt.xlabel(r'$y$',fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.ylabel(r'$y_{out}$',fontsize=30)
cbar.set_label('$log(density)$',fontsize=20)
cbar.ax.tick_params(labelsize=20)
cbar.solids.set_edgecolor("face")
plt.show()