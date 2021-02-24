# this file plots a png image of the solution at each timestep
# make a directory 'pngs' before running the script

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from params import x,y,x0,y0,t0,L,inv_couple,inv_w,inv_beta,w_reg
from synthetic_data import beta_true,w_bed,bed,w_lake

mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 2
mpl.rcParams['ytick.minor.width'] = 1


def plot_results(h,h_obs,w,beta,i):

    # normalizations for plotting (h,w,beta)
    h_max = np.max(h_obs)
    w_max = np.max(w_lake)
    beta_max = np.max(beta_true)

    # avoid dividing by zero in case w_true or beta_true is set to zero
    if w_max<1e-5:
        w_max = 1
    if beta_max<1e-5:
        beta_max = 1

    levels0 = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    levels = np.linspace(-1,1,9)

    ytix = [-4*L,-2*L,0,2*L,4*L]

    fig = plt.figure(figsize=(8,8))
    plt.suptitle(r'$t\, / \, T =$'+"{:.2f}".format(t0[i]/t0[-1]),fontsize=22 )
    plt.subplot(221)
    plt.annotate(r'$h$',xy=(-3.9*L,3*L),fontsize=18)
    plt.contourf(x0,y0,h[i,:,:].T/h_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.ylabel(r'$y$',fontsize=20)
    plt.gca().xaxis.set_ticklabels([])
    plt.yticks(ytix,fontsize=16)

    plt.subplot(222)
    plt.annotate(r'$h^{\mathrm{obs}}$',xy=(-3.9*L,2.9*L),fontsize=18)
    p1 = plt.contourf(x0,y0,h_obs[i,:,:].T/h_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.yticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().xaxis.set_ticklabels([])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.575, 0.03, 0.25])
    cbar = fig.colorbar(p1,cax=cbar_ax,orientation='vertical',ticks=levels0)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.tick_params(labelsize=18)


    plt.subplot(223)
    if inv_w == 1:
        plt.annotate(r'$w_b$',xy=(-3.9*L,3.15*L),fontsize=18)
        p2 = plt.contourf(x0,y0,w[i,:,:].T/w_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    elif inv_beta == 1:
        plt.annotate(r'$\beta$',xy=(-3.9*L,3.15*L),fontsize=18)
        p2 = plt.contourf(x0,y0,beta[i,:,:].T/beta_max,cmap='Blues',vmin=0,vmax=1,levels=np.linspace(0,1,9),extend='both')

    plt.ylabel(r'$y$',fontsize=20)
    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(ytix,fontsize=16)


    plt.subplot(224)
    if inv_w == 1:
        plt.annotate(r'$w_b^{\mathrm{true}}$',xy=(-3.9*L,2.9*L),fontsize=18)
        p2 = plt.contourf(x0,y0,w_lake[i,:,:].T/w_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    elif inv_beta == 1:
        plt.annotate(r'$\beta^{\mathrm{true}}$',xy=(-3.9*L,2.9*L),fontsize=18)
        p2 = plt.contourf(x0,y0,beta_true[i,:,:].T/beta_max,cmap='Blues',vmin=0,vmax=1,levels=np.linspace(0,1,9),extend='both')

    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])


    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.15, 0.03, 0.25])
    if inv_w == 1:
        cbar = fig.colorbar(p2,cax=cbar_ax,orientation='vertical',ticks=levels0)
    elif inv_beta == 1:
        cbar = fig.colorbar(p2,cax=cbar_ax,orientation='vertical',ticks=[0,0.25,0.5,0.75,1])

    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.tick_params(labelsize=18)

    plt.savefig('pngs/'+str(i),bbox_inches='tight')
    plt.close()
