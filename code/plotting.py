# this file includes various scripts for plotting the solutions

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from params import x,y,x0,y0,t0,L,H,Nt,Nx,t_final
from aux import nonlin_ex
import os

mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 2
mpl.rcParams['ytick.minor.width'] = 1


def plot_movie(data,fwd,sol,sol_true,inv_w,inv_beta):
    #* save a png image of the solution at each timestep
    #* need to make a directory called 'pngs' first!

    dim = inv_w + inv_beta

    if os.path.isdir('pngs')==False:
        os.mkdir('pngs')    # make a directory for the results.

    for i in range(Nt):
        if dim == 1 and nonlin_ex != 1:
            plot(sol,sol_true,fwd,data[0],i,inv_w,inv_beta)
        elif dim ==1 and nonlin_ex == 1:
            plot_1D(sol,sol_true,fwd,data,i)
        elif dim == 2:
            plot_joint(sol[0],sol_true[0],sol[1],sol_true[1],fwd[0],data[0],fwd[1],data[1],fwd[2],data[2],i)


def plot(sol,sol_true,h,h_obs,i,inv_w,inv_beta):

    # normalizations for plotting (h,w,beta)
    h_max = np.max(np.abs(h_obs))
    sol_max = np.max(np.abs(sol_true))

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
    p2 = plt.contourf(x0,y0,sol_true[i,:,:].T/sol_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    if inv_w == 1:
        plt.annotate(r'$w_b^{\mathrm{true}}$',xy=(-3.9*L,2.9*L),fontsize=18)

    elif inv_beta == 1:
        plt.annotate(r'$\beta^{\mathrm{true}}$',xy=(-3.9*L,2.9*L),fontsize=18)

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

def plot_joint(w,w_true,beta,beta_true,h,h_obs,u,u_obs,v,v_obs,i):

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
def snapshots(data,fwd,sol,sol_true,inv_w,inv_beta,dV_inv=None,dV_true=None):

    dim = inv_w + inv_beta
    if nonlin_ex == 1 and dim==1:
        snapshots_1D(data,fwd,sol,sol_true,dV_inv,dV_true)
    else:
        Nt0 = Nt
        levels0 = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
        levels = np.linspace(-1,1,9)
        h_obs = data[0]
        u = data[1]
        v = data[2]

        if dim==1:
            sol_max = np.max(np.abs(sol_true))
            sol1 = sol/sol_max
            sol2 = sol_true/sol_max
            alpha0 = 0.5

        elif dim==2:
            sol1 = sol[0]/np.max(np.abs(sol_true[0]))
            sol2 = sol[1]/np.max(np.abs(sol_true[1]))
            alpha0 = 0.75

        h_obs = h_obs/np.max(np.abs(h_obs))

        ds = int(7.5*Nx/100)                        # spacing for velocity plots

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

        p2 = plt.contourf(x0,y0,sol1[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

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

        p2 = plt.contourf(x0,y0,sol1[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

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

        p2 = plt.contourf(x0,y0,sol1[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

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

        p2 = plt.contourf(x0,y0,sol2[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

        plt.ylabel(r'$y$',fontsize=20)
        plt.xlabel(r'$x$',fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(ytix,fontsize=16)

        plt.subplot(3,3,8)
        plt.annotate(r'(h)',xy=(-38,30.5),fontsize=16,bbox=dict(facecolor='w',alpha=1))

        i = int(Nt0/2)

        p2 = plt.contourf(x0,y0,sol2[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

        plt.xlabel(r'$x$',fontsize=20)
        plt.xticks(fontsize=16)
        plt.gca().yaxis.set_ticklabels([])

        plt.subplot(3,3,9)
        plt.annotate(r'(i)',xy=(-38,30.5),fontsize=16,bbox=dict(facecolor='w',alpha=1))
        i = int(3*Nt0/4)

        p2 = plt.contourf(x0,y0,sol2[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

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

        if inv_w == 1 and dim == 1:
            plt.savefig('fig4',bbox_inches='tight')
        elif inv_beta == 1 and dim == 1:
            plt.savefig('fig5',bbox_inches='tight')
        else:
            plt.savefig('fig7',bbox_inches='tight')
        plt.show()
        plt.close()

#-------------------------------------------------------------------------------
def gps_plot(sol1,sol2,sol3,sol_true,vel_locs,inv_w,inv_beta):
    dim = inv_w + inv_beta

    Nt0 = Nt

    levels0 = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    levels = np.linspace(-1,1,9)

    sol_max = np.max(np.abs(sol_true))

    sol1 = sol1/sol_max
    sol2 = sol2/sol_max
    sol3 = sol3/sol_max

    ds = 10            # spacing for velocity plots


    ytix = np.array([-4*L,-2*L,0,2*L,4*L])


    statx = x[0,:,:][vel_locs[0,:,:]>1e-2]
    staty = y[0,:,:][vel_locs[0,:,:]>1e-2]

    X0,Y0 = np.meshgrid(statx,staty)

    fig = plt.figure(figsize=(10,7.5))


    # Col 1---------------------------------------------------------------------
    i = int(Nt0/4)

    plt.subplot(331)
    plt.annotate(r'(a)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    plt.title(r'$t\,/\,T = 0.25$',fontsize=22 )
    plt.ylabel(r'$y$',fontsize=20)
    plt.xticks(ytix,fontsize=16)
    plt.gca().xaxis.set_ticklabels([])
    p1 = plt.contourf(x0,y0,sol1[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.yticks(ytix,fontsize=16)
    ax2 = plt.gca().twinx()
    ax2.set_ylim(-40,40)
    ax2.set_xlim(-40,40)
    plt.yticks([])


    # Col 2---------------------------------------------------------------------
    i = int(Nt0/2)
    plt.subplot(332)
    plt.annotate(r'(b)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    plt.title(r'$t\,/\,T = 0.5$',fontsize=22 )
    plt.xticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().xaxis.set_ticklabels([])
    p1 = plt.contourf(x0,y0,sol1[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    # Col 3---------------------------------------------------------------------
    i = int(3*Nt0/4)

    plt.subplot(333)
    plt.annotate(r'(c)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    plt.title(r'$t\,/\,T = 0.75$',fontsize=22 )
    p1 = plt.contourf(x0,y0,sol1[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.xticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().xaxis.set_ticklabels([])


    # Col 1---------------------------------------------------------------------
    i = int(Nt0/4)

    plt.subplot(334)
    plt.annotate(r'(d)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    plt.ylabel(r'$y$',fontsize=20)
    plt.xticks(ytix,fontsize=16)
    plt.gca().xaxis.set_ticklabels([])
    p1 = plt.contourf(x0,y0,sol2[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.plot(np.array([0.0]),np.array([0.0]),'^',color='k',markersize=6,fillstyle='none')
    plt.yticks(ytix,fontsize=16)
    ax2 = plt.gca().twinx()
    ax2.set_ylim(-40,40)
    ax2.set_xlim(-40,40)
    plt.yticks([])


    # Col 2---------------------------------------------------------------------
    i = int(Nt0/2)
    plt.subplot(335)
    plt.annotate(r'(e)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    plt.xticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().xaxis.set_ticklabels([])
    p1 = plt.contourf(x0,y0,sol2[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.plot(np.array([0.0]),np.array([0.0]),'^',color='k',markersize=6,fillstyle='none')

    # Col 3---------------------------------------------------------------------
    i = int(3*Nt0/4)

    plt.subplot(336)
    plt.annotate(r'(f)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    p1 = plt.contourf(x0,y0,sol2[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.plot(np.array([0.0]),np.array([0.0]),'^',color='k',markersize=6,fillstyle='none')
    plt.xticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().xaxis.set_ticklabels([])


    # Col 1---------------------------------------------------------------------
    plt.subplot(337)
    plt.annotate(r'(g)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    plt.ylabel(r'$y$',fontsize=20)
    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(ytix,fontsize=16)

    p1 = plt.contourf(x0,y0,sol3[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.plot(X0.flatten(),Y0.flatten(),'^',color='k',markersize=6,fillstyle='none')

    plt.yticks(ytix,fontsize=16)

    ax2 = plt.gca().twinx()
    ax2.set_ylim(-40,40)
    ax2.set_xlim(-40,40)
    plt.yticks([])


    # Col 2---------------------------------------------------------------------
    i = int(Nt0/2)
    plt.subplot(338)
    plt.annotate(r'(h)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])

    p1 = plt.contourf(x0,y0,sol3[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.plot(X0.flatten(),Y0.flatten(),'k^',markersize=6,fillstyle='none')

    # Col 3---------------------------------------------------------------------
    i = int(3*Nt0/4)

    plt.subplot(339)
    plt.annotate(r'(i)',xy=(-38,30),fontsize=16,bbox=dict(facecolor='w',alpha=1))
    p1 = plt.contourf(x0,y0,sol3[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    dots, = plt.plot(X0.flatten(),Y0.flatten(),'k^',markersize=6,fillstyle='none',label='GPS station')
    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.legend(loc=(0.9,-0.425),fontsize=14,edgecolor='k',fancybox=False,framealpha=1)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.11, 0.02, 0.75])
    cbar = fig.colorbar(p1,cax=cbar_ax,orientation='vertical',ticks=levels0)
    cbar.set_label(label=r'$\beta^{\mathrm{inv}}\,/\, \Vert \beta^{\mathrm{true}}\Vert_\infty$',size=24)
    cbar.ax.get_yaxis().labelpad = 10
    cbar.ax.tick_params(labelsize=18)

    plt.savefig('fig6',bbox_inches='tight')
    plt.show()
    plt.close()
#-------------------------------------------------------------------------------


def snapshots_1D(data,fwd,sol,sol_true,dV_inv,dV_true):

    h_max = np.max(np.abs(data[0]))
    sol_max = np.max(np.abs(sol_true))

    data = data[0]/h_max
    fwd = fwd/h_max

    sol1 = sol/sol_max
    sol2 = sol_true/sol_max

    plt.figure(figsize=(10,8))
    plt.subplot(311)
    plt.annotate(r'(a)',xy=(-0.042,1.05),fontsize=20,bbox=dict(facecolor='w',alpha=1))
    plt.plot(t0/t_final,dV_true,linewidth=3,color='royalblue',label=r'$\Delta V^\mathrm{true}$')
    plt.plot(t0/t_final,dV_inv,linewidth=3,color='k',linestyle='--',label=r'$\Delta V^\mathrm{inv}$')
    plt.annotate(r'$t_1$',xy=(t0[47]/t_final-0.025,dV_true[47]+4),fontsize=24)
    plt.annotate(r'$t_2$',xy=(t0[100]/t_final-0.025,dV_true[100]-7),fontsize=24)
    plt.annotate(r'$t_3$',xy=(t0[113]/t_final+0.008,dV_true[113]+3),fontsize=24)
    plt.plot(t0[47]/t_final,dV_true[47],'o',color='crimson',markersize=12)
    plt.plot(t0[100]/t_final,dV_true[100],'o',color='crimson',markersize=12)
    plt.plot(t0[113]/t_final,dV_true[113],'o',color='crimson',markersize=12)
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(-0.05,1)
    plt.legend(fontsize=18,bbox_to_anchor=(1.01,0.8))
    plt.xlabel(r'$t\,/\,T$', fontsize=20)
    plt.ylabel(r'$\Delta V$', fontsize=20)
    plt.tight_layout()


    i = 47
    plt.subplot(334)
    plt.title(r'$t_1$',fontsize=24 )
    plt.annotate(r'(b)',xy=(-42,0.875),fontsize=20,bbox=dict(facecolor='w',alpha=1))
    plt.plot(x0,data[i,:,50],color='royalblue',linewidth=3,label=r'$h^{\mathrm{obs}}$')
    plt.plot(x0,fwd[i,:,50],color='k',linestyle='--',linewidth=3,label=r'$h^{\mathrm{fwd}}$')
    plt.ylabel(r'$h \,/\,\Vert h^{\mathrm{obs}}\Vert_\infty$',fontsize=20)
    plt.yticks(fontsize=16)
    plt.ylim(-1.25,1.25)
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

    plt.subplot(337)
    plt.annotate(r'(e)',xy=(-42,0.885),fontsize=20,bbox=dict(facecolor='w',alpha=1))
    plt.plot(x0,sol2[i,:,50],color='royalblue',linewidth=3,label=r'$w_b^{\mathrm{true}}$')
    plt.plot(x0,sol1[i,:,50],color='k',linestyle='--',linewidth=3,label=r'$w_b^{\mathrm{inv}}$')
    plt.ylabel(r'$w_b\,/\,\Vert w_b^{\mathrm{true}}\Vert_\infty$',fontsize=20)
    plt.ylim(-1.25,1.25)
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)


    i = 100
    plt.subplot(335)
    plt.annotate(r'(c)',xy=(-42,0.875),fontsize=20,bbox=dict(facecolor='w',alpha=1))
    plt.title(r'$t_2$',fontsize=24 )
    plt.plot(x0,data[i,:,50],color='royalblue',linewidth=3,label=r'$h^{\mathrm{obs}}$')
    plt.plot(x0,fwd[i,:,50],color='k',linestyle='--',linewidth=3,label=r'$h^{\mathrm{fwd}}$')
    plt.ylim(-1.25,1.25)
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])

    plt.subplot(338)
    plt.annotate(r'(f)',xy=(-42,0.885),fontsize=20,bbox=dict(facecolor='w',alpha=1))
    plt.plot(x0,sol2[i,:,50],color='royalblue',linewidth=3,label=r'true sol.')
    plt.plot(x0,sol1[i,:,50],color='k',linestyle='--',linewidth=3,label=r'inversion')
    plt.ylim(-1.25,1.25)
    plt.gca().yaxis.set_ticklabels([])
    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)

    i = 113
    plt.subplot(336)
    plt.title(r'$t_3$',fontsize=24)
    plt.annotate(r'(d)',xy=(-42,0.875),fontsize=20,bbox=dict(facecolor='w',alpha=1))
    plt.plot(x0,data[i,:,50],color='royalblue',linewidth=3,label=r'$h^{\mathrm{obs}}$')
    plt.plot(x0,fwd[i,:,50],color='k',linestyle='--',linewidth=3,label=r'$h^{\mathrm{fwd}}$')
    plt.legend(fontsize=18,bbox_to_anchor=(1.01,0.8))
    plt.ylim(-1.25,1.25)
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])

    plt.subplot(339)
    plt.annotate(r'(g)',xy=(-42,0.885),fontsize=20,bbox=dict(facecolor='w',alpha=1))
    plt.plot(x0,sol2[i,:,50],color='royalblue',linewidth=3,label=r'$w_b^{\mathrm{true}}$')
    plt.plot(x0,sol1[i,:,50],color='k',linestyle='--',linewidth=3,label=r'$w_b^{\mathrm{inv}}$')
    plt.ylim(-1.25,1.25)
    plt.legend(fontsize=18,bbox_to_anchor=(1.01,0.8))
    plt.gca().yaxis.set_ticklabels([])
    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)

    plt.tight_layout()
    plt.savefig('fig9',bbox_inches='tight')
    plt.show()
    plt.close()


#-------------------------------------------------------------------------------
def plot_1D(sol,sol_true,fwd,data,i):

    h_obs = data[0]
    h_max = np.max(np.abs(h_obs))
    sol_max = np.max(np.abs(sol_true))

    data = h_obs/h_max
    fwd = fwd/h_max


    sol1 = sol/sol_max
    sol2 = sol_true/sol_max


    plt.figure(figsize=(8,10))
    plt.suptitle(r'$t\, / \, T =$'+"{:.2f}".format(t0[i]/t0[-1]),fontsize=22 )
    plt.subplot(211)
    plt.plot(x0,data[i,:,50],color='royalblue',linewidth=3,label=r'$h^{\mathrm{obs}}$')
    plt.plot(x0,fwd[i,:,50],color='k',linestyle='--',linewidth=3,label=r'$h^{\mathrm{fwd}}$')

    plt.ylabel(r'$h \,/\,\Vert h^{\mathrm{obs}}\Vert_\infty$',fontsize=20)
    plt.yticks(fontsize=16)
    plt.ylim(-1.25,1.25)
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.legend(fontsize=20,loc='lower right')

    plt.subplot(212)

    plt.plot(x0,sol2[i,:,50],color='royalblue',linewidth=3,label=r'$w_b^{\mathrm{true}}$')
    plt.plot(x0,sol1[i,:,50],color='k',linestyle='--',linewidth=3,label=r'$w_b^\mathrm{inv}$')

    plt.ylabel(r'$w_b\,/\,\Vert w_b^{\mathrm{true}}\Vert_\infty$',fontsize=20)
    plt.ylim(-1.25,1.25)
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    plt.legend(fontsize=20,loc='lower right')

    # Label axes and save png:
    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig('pngs/'+str(i))
    plt.close()
