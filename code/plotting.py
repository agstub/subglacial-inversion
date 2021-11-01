# this file includes various scripts for plotting the solutions

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from params import x,y,x0,y0,t0,L,inv_w,inv_beta,H,Nt,dim,Nx,vel_data
from gps_stats import gps
from synthetic_data import beta_true,w_true,u_true,v_true,sigma,s_true
import os

mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 2
mpl.rcParams['ytick.minor.width'] = 1


def plot_movie(sol,fwd,data):
    #* save a png image of the solution at each timestep
    #* need to make a directory called 'pngs' first!
    if os.path.isdir('pngs')==False:
        os.mkdir('pngs')    # make a directory for the results.

    for i in range(Nt):
        if dim == 1 :
            plot(sol,fwd,data,i)
        elif dim == 2:
            plot_joint(sol[0],sol[1],fwd[0],data[0],fwd[1],data[1],fwd[2],data[2],i)


def plot(sol,h,h_obs,i):

    # normalizations for plotting (h,w,beta)
    sol_true = beta_true+w_true+m_true
    h_max = np.max([np.max(np.abs(h_obs)),1e-10])
    sol_max = np.max([np.max(np.abs(sol_true)),1e-10])


    levels0 = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    levels = np.linspace(-1,1,9)

    ytix = [-4*L,-2*L,0,2*L,4*L]

    fig = plt.figure(figsize=(8,8))
    plt.suptitle(r'$t\, / \, T =$'+"{:.2f}".format(t0[i]/t0[-1]),fontsize=22 )
    plt.subplot(221)
    plt.annotate(r'$h^\mathrm{fwd}$',xy=(-3.9*L,3*L),fontsize=18)
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
    cbar_ax = fig.add_axes([0.875, 0.575, 0.02, 0.25])
    cbar = fig.colorbar(p1,cax=cbar_ax,orientation='vertical',ticks=levels0)
    cbar.set_label(label=r'$h\,/\, \Vert h^\mathrm{obs}\Vert_\infty$',size=18)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.tick_params(labelsize=18)


    plt.subplot(223)
    if inv_w == 1:
        plt.annotate(r'$w_b^\mathrm{inv}$',xy=(-3.9*L,3.15*L),fontsize=18)
        p2 = plt.contourf(x0,y0,sol[i,:,:].T/sol_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    elif inv_beta == 1:
        plt.annotate(r'$\beta^\mathrm{inv}$',xy=(-3.9*L,3.15*L),fontsize=18)
        p2 = plt.contourf(x0,y0,sol[i,:,:].T/sol_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.ylabel(r'$y$',fontsize=20)
    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(ytix,fontsize=16)


    plt.subplot(224)
    if inv_w == 1:
        plt.annotate(r'$w_b^{\mathrm{true}}$',xy=(-3.9*L,2.9*L),fontsize=18)
        p2 = plt.contourf(x0,y0,w_true[i,:,:].T/sol_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    elif inv_beta == 1:
        plt.annotate(r'$\beta^{\mathrm{true}}$',xy=(-3.9*L,2.9*L),fontsize=18)
        p2 = plt.contourf(x0,y0,beta_true[i,:,:].T/sol_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])


    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.15, 0.02, 0.25])

    cbar = fig.colorbar(p2,cax=cbar_ax,orientation='vertical',ticks=levels0)

    if inv_w == 1:
        cbar.set_label(label=r'$w_b\,/\, \Vert w_b^\mathrm{true}\Vert_\infty$',size=18)
    elif inv_beta == 1:
        cbar.set_label(label=r'$\beta\,/\, \Vert \beta^\mathrm{true}\Vert_\infty$',size=18)

    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.tick_params(labelsize=18)

    plt.savefig('pngs/'+str(i),bbox_inches='tight')
    plt.close()


#-------------------------------------------------------------------------------

def plot_joint(w,beta,h,h_obs,u,u_obs,v,v_obs,i):

    # normalizations for plotting (h,w,beta)
    h_max = np.max(np.abs(h_obs))
    w_max = np.max( [np.max(np.abs(w_true)),1e-10] )
    beta_max = np.max([np.max(np.abs(beta_true)),1e-10])

    ds = int(5*Nx/100)            # spacing for velocity plots

    u_max =  np.max(np.sqrt(u**2+v**2))

    v_max = np.max(np.sqrt(u**2+v**2))


    levels0 = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    levels = np.linspace(-1,1,9)

    ytix = [-4*L,-2*L,0,2*L,4*L]

    fig = plt.figure(figsize=(8,12))
    plt.suptitle(r'$t\, / \, T =$'+"{:.2f}".format(t0[i]/t0[-1]),fontsize=22 )
    plt.subplot(321)
    plt.annotate(r'$h^\mathrm{fwd}$, $\mathbf{u}^\mathrm{fwd}$',xy=(-3.9*L,3*L),fontsize=18)
    plt.contourf(x0,y0,h[i,:,:].T/h_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.quiver(x[0,::ds,::ds],y[0,::ds,::ds],u[i,::ds,::ds]/u_max,v[i,::ds,::ds]/v_max,units='inches',scale=4,minlength=0,width=0.02,color='k')

    plt.ylabel(r'$y$',fontsize=20)
    plt.gca().xaxis.set_ticklabels([])
    plt.yticks(ytix,fontsize=16)

    plt.subplot(322)
    plt.annotate(r'$h^{\mathrm{obs}}$, $\mathbf{u}^\mathrm{obs}$',xy=(-3.9*L,2.9*L),fontsize=18)
    p1 = plt.contourf(x0,y0,h_obs[i,:,:].T/h_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.quiver(x[0,::ds,::ds],y[0,::ds,::ds],u_obs[i,::ds,::ds]/u_max,v_obs[i,::ds,::ds]/v_max,units='inches',scale=4,minlength=0,width=0.02,color='k')


    plt.yticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().xaxis.set_ticklabels([])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.665, 0.03, 0.2])
    cbar = fig.colorbar(p1,cax=cbar_ax,orientation='vertical',ticks=levels0)
    cbar.set_label(label=r'$h\,/\, \Vert h^\mathrm{obs}\Vert_\infty$',size=18)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.tick_params(labelsize=18)


    plt.subplot(323)

    plt.annotate(r'$w_b^\mathrm{inv}$',xy=(-3.9*L,3.15*L),fontsize=18)
    p2 = plt.contourf(x0,y0,w[i,:,:].T/w_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.ylabel(r'$y$',fontsize=20)
    plt.gca().xaxis.set_ticklabels([])

    plt.yticks(ytix,fontsize=16)


    plt.subplot(324)
    plt.annotate(r'$w_b^{\mathrm{true}}$',xy=(-3.9*L,2.9*L),fontsize=18)
    p2 = plt.contourf(x0,y0,w_true[i,:,:].T/w_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.xticks(fontsize=16)
    plt.yticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().xaxis.set_ticklabels([])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.395, 0.03, 0.2])
    cbar = fig.colorbar(p2,cax=cbar_ax,orientation='vertical',ticks=levels0)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.tick_params(labelsize=18)


    cbar.set_label(label=r'$w_b\,/\, \Vert w_b^\mathrm{true}\Vert_\infty$',size=18)

    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.tick_params(labelsize=18)

    plt.subplot(325)

    plt.annotate(r'$\beta^\mathrm{inv}$',xy=(-3.9*L,3.15*L),fontsize=18)
    p3 = plt.contourf(x0,y0,beta[i,:,:].T/beta_max,cmap='coolwarm',vmin=-1,vmax=1,levels=np.linspace(-1,1,9),extend='both')


    plt.ylabel(r'$y$',fontsize=20)
    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(ytix,fontsize=16)


    plt.subplot(326)

    plt.annotate(r'$\beta^{\mathrm{true}}$',xy=(-3.9*L,2.9*L),fontsize=18)
    plt.contourf(x0,y0,beta_true[i,:,:].T/beta_max,cmap='coolwarm',vmin=-1,vmax=1,levels=np.linspace(-1,1,9),extend='both')


    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])


    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.12, 0.03, 0.2])

    cbar = fig.colorbar(p3,cax=cbar_ax,orientation='vertical',ticks=levels0)

    cbar.set_label(label=r'$\beta\,/\, \Vert \beta^\mathrm{true}\Vert_\infty$',size=18)

    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.tick_params(labelsize=18)


    plt.savefig('pngs/'+str(i),bbox_inches='tight')
    plt.close()





#-------------------------------------------------------------------------------
def snapshots(sol,data,sol_true):

    Nt0 = Nt

    levels0 = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    levels = np.linspace(-1,1,9)

    if vel_data == 0:
        h_obs = data
        u = u_true
        v = v_true
        sol_max = np.max(np.abs(sol_true))
        sol = sol/sol_max
        sol_true = sol_true/sol_max
        alpha0 = 0.5

    elif vel_data ==1 and dim==2:
        h_obs = data[0]
        u = data[1]
        v = data[2]
        sol = sol/np.max(np.abs(w_true))
        sol_true = sol_true/np.max(np.abs(beta_true))
        alpha0 = 0.75

    elif vel_data ==1 and dim==1:
        h_obs = data[0]
        u = data[1]
        v = data[2]
        sol_max = np.max(np.abs(sol_true))
        sol = sol/sol_max
        sol_true = sol_true/sol_max
        alpha0 = 0.75

    h_obs = h_obs/np.max(np.abs(h_obs))

    ds = int(7.5*Nx/100)             # spacing for velocity plots

    u_max = np.max(np.sqrt(u**2+v**2))
    v_max = np.max(np.sqrt(u**2+v**2))

    ytix = np.array([-4*L,-2*L,0,2*L,4*L])

    fig = plt.figure(figsize=(10,8))

    # Col 1---------------------------------------------------------------------
    i = int(Nt0/4)
    plt.subplot(331)
    plt.annotate(r'(a)',xy=(-38,30.5),fontsize=16,bbox=dict(facecolor='w',alpha=1))

    plt.title(r'$t\,/\,T = 0.25$',fontsize=22 )

    plt.ylabel(r'$y$',fontsize=20)

    p1 = plt.contourf(x0,y0,h_obs[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.quiver(x[0,::ds,::ds],y[0,::ds,::ds],u[i,::ds,::ds]/u_max,v[i,::ds,::ds]/v_max,units='inches',alpha=alpha0,scale=4,minlength=0,width=0.02,color='k')

    plt.yticks(ytix,fontsize=16)
    plt.gca().xaxis.set_ticklabels([])

    plt.subplot(334)
    plt.annotate(r'(d)',xy=(-38,30.5),fontsize=16,bbox=dict(facecolor='w',alpha=1))

    p2 = plt.contourf(x0,y0,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.ylabel(r'$y$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(ytix,fontsize=16)
    plt.gca().xaxis.set_ticklabels([])

    # Col 2---------------------------------------------------------------------
    i = int(Nt0/2)
    plt.subplot(332)
    plt.annotate(r'(b)',xy=(-38,30.5),fontsize=16,bbox=dict(facecolor='w',alpha=1))

    plt.title(r'$t\,/\,T = 0.5$',fontsize=22 )

    plt.gca().yaxis.set_ticklabels([])
    plt.gca().xaxis.set_ticklabels([])

    p1 = plt.contourf(x0,y0,h_obs[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.quiver(x[0,::ds,::ds],y[0,::ds,::ds],u[i,::ds,::ds]/u_max,v[i,::ds,::ds]/v_max,units='inches',alpha=alpha0,scale=4,minlength=0,width=0.02,color='k')

    plt.gca().xaxis.set_ticklabels([])

    plt.subplot(335)
    plt.annotate(r'(e)',xy=(-38,30.5),fontsize=16,bbox=dict(facecolor='w',alpha=1))

    plt.xticks(fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().xaxis.set_ticklabels([])

    p2 = plt.contourf(x0,y0,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    # Col 3---------------------------------------------------------------------
    i = int(3*Nt0/4)
    plt.subplot(333)
    plt.annotate(r'(c)',xy=(-38,30.5),fontsize=16,bbox=dict(facecolor='w',alpha=1))

    plt.title(r'$t\,/\,T = 0.75$',fontsize=22 )

    p1 = plt.contourf(x0,y0,h_obs[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.quiver(x[0,::ds,::ds],y[0,::ds,::ds],u[i,::ds,::ds]/u_max,v[i,::ds,::ds]/v_max,units='inches',alpha=alpha0,scale=4,minlength=0,width=0.02,color='k')

    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.645, 0.02, 0.25])
    cbar = fig.colorbar(p1,cax=cbar_ax,orientation='vertical',ticks=levels0)
    cbar.set_label(label=r'$h^\mathrm{obs}\,/\, \Vert h^\mathrm{obs}\Vert_\infty$',size=18)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.tick_params(labelsize=18)


    plt.subplot(336)
    plt.annotate(r'(f)',xy=(-38,30.5),fontsize=16,bbox=dict(facecolor='w',alpha=1))

    p2 = plt.contourf(x0,y0,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.37, 0.02, 0.25])

    cbar = fig.colorbar(p2,cax=cbar_ax,orientation='vertical',ticks=levels)

    if inv_w==1 and dim==1:
        cbar.set_label(label=r'$w_b^\mathrm{inv}\,/\, \Vert w_b^\mathrm{true}\Vert_\infty$',size=18)
    elif inv_beta==1 and dim==1:
        cbar.set_label(label=r'$\beta^\mathrm{inv}\,/\, \Vert \beta^\mathrm{true}\Vert_\infty$',size=18)
    elif dim == 2:
        cbar.set_label(label=r'$w_b^\mathrm{inv}\,/\, \Vert w_b^\mathrm{true}\Vert_\infty$',size=18)


    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.tick_params(labelsize=18)



    plt.subplot(3,3,7)

    i = int(Nt0/4)
    plt.annotate(r'(g)',xy=(-38,30.5),fontsize=16,bbox=dict(facecolor='w',alpha=1))

    p2 = plt.contourf(x0,y0,sol_true[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.ylabel(r'$y$',fontsize=20)
    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(ytix,fontsize=16)

    plt.subplot(3,3,8)
    plt.annotate(r'(h)',xy=(-38,30.5),fontsize=16,bbox=dict(facecolor='w',alpha=1))

    i = int(Nt0/2)

    p2 = plt.contourf(x0,y0,sol_true[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.gca().yaxis.set_ticklabels([])

    plt.subplot(3,3,9)
    plt.annotate(r'(i)',xy=(-38,30.5),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    i = int(3*Nt0/4)

    p2 = plt.contourf(x0,y0,sol_true[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.gca().yaxis.set_ticklabels([])


    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.095, 0.02, 0.25])

    cbar = fig.colorbar(p2,cax=cbar_ax,orientation='vertical',ticks=levels)

    if inv_w ==1 and dim==1:
        cbar.set_label(label=r'$w_b^\mathrm{true}\,/\, \Vert w_b^\mathrm{true}\Vert_\infty$',size=18)
    elif inv_beta ==1 and dim==1:
        cbar.set_label(label=r'$\beta^\mathrm{true}\,/\, \Vert \beta^\mathrm{true}\Vert_\infty$',size=18)
    elif dim == 2:
        cbar.set_label(label=r'$\beta^\mathrm{inv}\,/\, \Vert \beta^\mathrm{true}\Vert_\infty$',size=18)


    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.tick_params(labelsize=18)

    plt.savefig('snaps',bbox_inches='tight')
    plt.close()



#-------------------------------------------------------------------------------
def gps_plot(sol,sol2,sol_true):

    Nt0 = Nt

    levels0 = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    levels = np.linspace(-1,1,9)

    sol_max = np.max(np.abs(sol_true))

    sol = sol/sol_max
    sol2 = sol2/sol_max

    ds = 10            # spacing for velocity plots


    ytix = np.array([-4*L,-2*L,0,2*L,4*L])


    statx = x[0,:,:][gps()[0,:,:]>1e-2]
    staty = y[0,:,:][gps()[0,:,:]>1e-2]

    X0,Y0 = np.meshgrid(statx,staty)

    fig = plt.figure(figsize=(10,5))


    # Col 1---------------------------------------------------------------------
    i = int(Nt0/4)

    plt.subplot(231)
    plt.annotate(r'(a)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    plt.title(r'$t\,/\,T = 0.25$',fontsize=22 )
    plt.ylabel(r'$y$',fontsize=20)
    plt.xticks(ytix,fontsize=16)
    plt.gca().xaxis.set_ticklabels([])

    p1 = plt.contourf(x0,y0,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.plot(np.array([0.0]),np.array([0.0]),'^',color='k',markersize=6,fillstyle='none')

    plt.yticks(ytix,fontsize=16)

    ax2 = plt.gca().twinx()
    ax2.set_ylim(-40,40)
    ax2.set_xlim(-40,40)
    plt.yticks([])


    # Col 2---------------------------------------------------------------------
    i = int(Nt0/2)
    plt.subplot(232)
    plt.annotate(r'(b)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    plt.title(r'$t\,/\,T = 0.5$',fontsize=22 )
    plt.xticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().xaxis.set_ticklabels([])

    p1 = plt.contourf(x0,y0,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.plot(np.array([0.0]),np.array([0.0]),'^',color='k',markersize=6,fillstyle='none')

    # Col 3---------------------------------------------------------------------
    i = int(3*Nt0/4)

    plt.subplot(233)
    plt.annotate(r'(c)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    plt.title(r'$t\,/\,T = 0.75$',fontsize=22 )

    p1 = plt.contourf(x0,y0,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.plot(np.array([0.0]),np.array([0.0]),'^',color='k',markersize=6,fillstyle='none')

    plt.xticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().xaxis.set_ticklabels([])


    plt.subplot(234)
    plt.annotate(r'(d)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    plt.ylabel(r'$y$',fontsize=20)
    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(ytix,fontsize=16)

    p1 = plt.contourf(x0,y0,sol2[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.plot(X0.flatten(),Y0.flatten(),'^',color='k',markersize=6,fillstyle='none')

    plt.yticks(ytix,fontsize=16)

    ax2 = plt.gca().twinx()
    ax2.set_ylim(-40,40)
    ax2.set_xlim(-40,40)
    plt.yticks([])


    # Col 2---------------------------------------------------------------------
    i = int(Nt0/2)
    plt.subplot(235)
    plt.annotate(r'(e)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])

    p1 = plt.contourf(x0,y0,sol2[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.plot(X0.flatten(),Y0.flatten(),'k^',markersize=6,fillstyle='none')

    # Col 3---------------------------------------------------------------------
    i = int(3*Nt0/4)

    plt.subplot(236)
    plt.annotate(r'(f)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    p1 = plt.contourf(x0,y0,sol2[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    dots, = plt.plot(X0.flatten(),Y0.flatten(),'k^',markersize=6,fillstyle='none',label='GPS station')

    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.legend(loc=(0.9,-0.425),fontsize=14,edgecolor='k',fancybox=False,framealpha=1)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.11, 0.02, 0.75])
    cbar = fig.colorbar(p1,cax=cbar_ax,orientation='vertical',ticks=levels0)

    cbar.set_label(label=r'$\beta^{\mathrm{inv}}\,/\, \Vert \beta^{\mathrm{true}}\Vert_\infty$',size=22)

    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.tick_params(labelsize=18)

    plt.savefig('gps',bbox_inches='tight')
    plt.close()
#-------------------------------------------------------------------------------


# discrepancy diagram plot
def discrepancy(mis1,eps1,mis2,eps2):

    plt.figure(figsize=(8,6))
    plt.axhline(y=1e-2,linestyle='--',color='r',linewidth=2)
    plt.plot(eps2[mis2>1e-7],mis2[mis2>1e-7],'o-',linewidth=2,label=r'$\beta$')
    plt.plot([3.319e1],[1e-2],'*',color='k',markersize=16)
    plt.plot(eps1[mis1>1e-7],mis1[mis1>1e-7],'s-',linewidth=2,label=r'$w_b$')
    plt.plot([4.27e-5],[1e-2],'*',color='k',markersize=16)
    plt.annotate(r'noise level',xy=(5e-2,1.1e-2),fontsize=16,color='r')
    plt.xlabel(r'$\varepsilon$',fontsize=20)
    plt.ylabel(r'$\Vert h^\varepsilon-h^{\mathrm{obs}}\Vert\, / \, \Vert h^\mathrm{obs}\Vert$',fontsize=20)
    plt.legend(fontsize=20,loc='upper right')
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig('Lcurve')
    plt.close()
