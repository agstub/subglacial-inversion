# this file plots a png image of the solution at each timestep
# *make a directory 'pngs' before running the script

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from params import x,y,x0,y0,t0,L,inv_w,inv_beta,inv_m,H,Nt,dim,Nx
from synthetic_data import beta_true,w_true,m_true,u_true,v_true
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
    h_max = np.max([np.max(np.abs(h_obs)),1e-10])
    sol_max = np.max([np.max(np.abs(sol)),1e-10])


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
        p2 = plt.contourf(x0,y0,sol[i,:,:].T/sol_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    elif inv_beta == 1:
        plt.annotate(r'$\beta$',xy=(-3.9*L,3.15*L),fontsize=18)
        p2 = plt.contourf(x0,y0,sol[i,:,:].T/sol_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    elif inv_m == 1:
        plt.annotate(r'$m$',xy=(-3.9*L,3.15*L),fontsize=18)
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

    elif inv_m == 1:
        plt.annotate(r'$m^{\mathrm{true}}$',xy=(-3.9*L,2.9*L),fontsize=18)
        p2 = plt.contourf(x0,y0,m_true[i,:,:].T/sol_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.xlabel(r'$x$',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])


    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.15, 0.03, 0.25])
    if inv_w == 1 or inv_m==1:
        cbar = fig.colorbar(p2,cax=cbar_ax,orientation='vertical',ticks=levels0)
    elif inv_beta == 1:
        cbar = fig.colorbar(p2,cax=cbar_ax,orientation='vertical',ticks=[0,0.25,0.5,0.75,1])

    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.tick_params(labelsize=18)

    plt.savefig('pngs/'+str(i),bbox_inches='tight')
    plt.close()


#-------------------------------------------------------------------------------

def plot_joint(w,beta,h,h_obs,u,u_obs,v,v_obs,i):

    # normalizations for plotting (h,w,beta)
    h_max = np.max(np.abs(h_obs))
    w_max = np.max( [np.max(np.abs(w)),1e-10] )
    beta_max = np.max([np.max(np.abs(beta)),1e-10])

    ds = int(10*Nx/100)            # spacing for velocity plots

    u_max = np.max(np.abs(u_obs))
    v_max = np.max(np.abs(v_obs))


    levels0 = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    levels = np.linspace(-1,1,9)

    ytix = [-4*L,-2*L,0,2*L,4*L]

    fig = plt.figure(figsize=(8,12))
    plt.suptitle(r'$t\, / \, T =$'+"{:.2f}".format(t0[i]/t0[-1]),fontsize=22 )
    plt.subplot(321)
    plt.annotate(r'$h$, $\mathbf{u}$',xy=(-3.9*L,3*L),fontsize=18)
    plt.contourf(x0,y0,h[i,:,:].T/h_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.quiver(x[0,::ds,::ds],y[0,::ds,::ds],u[i,::ds,::ds]/u_max,v[i,::ds,::ds]/v_max,units='inches',scale=3,color='k')

    plt.ylabel(r'$y$',fontsize=20)
    plt.gca().xaxis.set_ticklabels([])
    plt.yticks(ytix,fontsize=16)

    plt.subplot(322)
    plt.annotate(r'$h^{\mathrm{obs}}$, $\mathbf{u}^\mathrm{obs}$',xy=(-3.9*L,2.9*L),fontsize=18)
    p1 = plt.contourf(x0,y0,h_obs[i,:,:].T/h_max,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    plt.quiver(x[0,::ds,::ds],y[0,::ds,::ds],u_obs[i,::ds,::ds]/u_max,v_obs[i,::ds,::ds]/v_max,units='inches',scale=3,color='k')


    plt.yticks(ytix,fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().xaxis.set_ticklabels([])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.665, 0.03, 0.2])
    cbar = fig.colorbar(p1,cax=cbar_ax,orientation='vertical',ticks=levels0)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.tick_params(labelsize=18)


    plt.subplot(323)

    plt.annotate(r'$w_b$',xy=(-3.9*L,3.15*L),fontsize=18)
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

    plt.subplot(325)

    plt.annotate(r'$\beta$',xy=(-3.9*L,3.15*L),fontsize=18)
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

    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.tick_params(labelsize=18)

    plt.savefig('pngs/'+str(i),bbox_inches='tight')
    plt.close()




def snapshots(sol,h_obs):

    xy_sc = 1

    levels0 = [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]
    levels = np.linspace(-1,1,9)
    levels_m = np.linspace(-1,1,9)

    h_obs = h_obs/np.max(np.abs(h_obs))
    sol = sol/np.abs(np.max(sol))

    ds = 10            # spacing for velocity plots

    u_max = np.max(np.abs(u_true))
    v_max = np.max(np.abs(v_true))

    ytix = np.array([-4*L,-2*L,0,2*L,4*L])*xy_sc

    fig = plt.figure(figsize=(13,8))

    # Col 4---------------------------------------------------------------------
    i = int(3*Nt/4)
    plt.subplot(244)
    plt.title(r'$t =$'+"{:.1f}".format(t0[i])+' yr',fontsize=22 )

    p1 = plt.contourf(x0*xy_sc,y0*xy_sc,h_obs[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.quiver(xy_sc*x[0,::ds,::ds],xy_sc*y[0,::ds,::ds],u_true[i,::ds,::ds]/u_max,v_true[i,::ds,::ds]/v_max,units='inches',scale=3,color='k')

    plt.yticks(ytix,fontsize=16)
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.55, 0.02, 0.3])
    cbar = fig.colorbar(p1,cax=cbar_ax,orientation='vertical',ticks=levels0)
    cbar.set_label(label=r'elevation anomaly (m)',size=18)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.tick_params(labelsize=18)

    plt.subplot(248)
    if inv_m == 1:
        p2 = plt.contourf(x0*xy_sc,y0*xy_sc,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels_m,extend='both')


    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.gca().yaxis.set_ticklabels([])



    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.875, 0.125, 0.02, 0.3])
    if inv_w == 1 or inv_m==1:
        cbar = fig.colorbar(p2,cax=cbar_ax,orientation='vertical',ticks=levels_m)
        cbar.set_label(label=r'melt rate (m/yr)',size=18)
    elif inv_beta == 1:
        cbar = fig.colorbar(p2,cax=cbar_ax,orientation='vertical',ticks=[0,0.25,0.5,0.75,1])

    cbar.ax.get_yaxis().labelpad = 40
    cbar.ax.tick_params(labelsize=18)

    # Col 1---------------------------------------------------------------------
    i = 0
    plt.subplot(241)
    plt.title(r'$t =$'+"{:.1f}".format(t0[i])+' yr',fontsize=22 )
    plt.ylabel(r'$y$ (km)',fontsize=20)


    p1 = plt.contourf(x0*xy_sc,y0*xy_sc,h_obs[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.quiver(xy_sc*x[0,::ds,::ds],xy_sc*y[0,::ds,::ds],u_true[i,::ds,::ds]/u_max,v_true[i,::ds,::ds]/v_max,units='inches',scale=3,color='k')


    plt.yticks(ytix,fontsize=16)
    plt.gca().xaxis.set_ticklabels([])

    plt.subplot(245)
    if inv_w == 1:
        plt.annotate(r'$w_b$',xy=(-3.9*L,3.15*L),fontsize=18)
        p2 = plt.contourf(x0*xy_sc,y0*xy_sc,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    elif inv_beta == 1:
        plt.annotate(r'$\beta$',xy=(-3.9*L,3.15*L),fontsize=18)
        p2 = plt.contourf(x0*xy_sc,y0*xy_sc,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    elif inv_m == 1:
        p2 = plt.contourf(x0*xy_sc,y0*xy_sc,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels_m,extend='both')

    plt.ylabel(r'$y$ (km)',fontsize=20)
    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(ytix,fontsize=16)

    # Col 2---------------------------------------------------------------------
    i = int(Nt/4)
    plt.subplot(242)
    plt.title(r'$t =$'+"{:.1f}".format(t0[i])+' yr',fontsize=22 )
    plt.gca().yaxis.set_ticklabels([])
    plt.gca().xaxis.set_ticklabels([])


    p1 = plt.contourf(x0*xy_sc,y0*xy_sc,h_obs[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.quiver(xy_sc*x[0,::ds,::ds],xy_sc*y[0,::ds,::ds],u_true[i,::ds,::ds]/u_max,v_true[i,::ds,::ds]/v_max,units='inches',scale=3,color='k')


    plt.gca().xaxis.set_ticklabels([])

    plt.subplot(246)
    plt.xticks(fontsize=16)
    plt.gca().yaxis.set_ticklabels([])
    plt.xlabel(r'$x$ (km)',fontsize=20)

    if inv_w == 1:
        plt.annotate(r'$w_b$',xy=(-3.9*L,3.15*L),fontsize=18)
        p2 = plt.contourf(x0*xy_sc,y0*xy_sc,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    elif inv_beta == 1:
        plt.annotate(r'$\beta$',xy=(-3.9*L,3.15*L),fontsize=18)
        p2 = plt.contourf(x0,y0,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')

    elif inv_m == 1:
        p2 = plt.contourf(x0*xy_sc,y0*xy_sc,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels_m,extend='both')


    # Col 3---------------------------------------------------------------------
    i = int(Nt/2)
    plt.subplot(243)
    plt.title(r'$t =$'+"{:.1f}".format(t0[i])+' yr',fontsize=22 )

    p1 = plt.contourf(x0*xy_sc,y0*xy_sc,h_obs[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    plt.quiver(xy_sc*x[0,::ds,::ds],xy_sc*y[0,::ds,::ds],u_true[i,::ds,::ds]/u_max,v_true[i,::ds,::ds]/v_max,units='inches',scale=3,color='k')


    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])


    plt.subplot(247)
    if inv_w == 1:
        plt.annotate(r'$w_b$',xy=(-3.9*L,3.15*L),fontsize=18)
        p2 = plt.contourf(x0,y0,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels,extend='both')
    elif inv_beta == 1:
        plt.annotate(r'$\beta$',xy=(-3.9*L,3.15*L),fontsize=18)
        p2 = plt.contourf(x0,y0,sol[i,:,:].T,cmap='coolwarm',vmin=0,vmax=1,levels=levels,extend='both')

    elif inv_m == 1:
        p2 = plt.contourf(x0*xy_sc,y0*xy_sc,sol[i,:,:].T,cmap='coolwarm',vmin=-1,vmax=1,levels=levels_m,extend='both')

    plt.xlabel(r'$x$ (km)',fontsize=20)
    plt.xticks(fontsize=16)
    plt.gca().yaxis.set_ticklabels([])




    plt.savefig('snaps',bbox_inches='tight')
    plt.close()
