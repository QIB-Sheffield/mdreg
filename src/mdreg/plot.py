import os
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numpy as np


def animation(array, path=None, filename='animation', vmin=None, vmax=None, interval=250, show=False):

    array[np.isnan(array)] = 0
    shape = np.shape(array)

    if len(shape)==4: ##save 3D data
        for k in range(shape[2]): 
            fig, ax = plt.subplots()
            im = ax.imshow(array[:,:,k,0].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
            ims = []
            for i in range(shape[-1]):
                im = ax.imshow(array[:,:,k,i].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax) 
                ims.append([im]) 
            anim = ArtistAnimation(fig, ims, interval=interval)
            if path is not None:
                file_3D_save = os.path.join(path, filename)
                anim.save(file_3D_save + '_' + str(k) + ".gif")
            if show:
                plt.show()
            else:
                plt.close()

    else: # save 2D data  
        fig, ax = plt.subplots()
        im = ax.imshow(array[:,:,0].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
        ims = []
        for i in range(shape[-1]):
            im = ax.imshow(array[:,:,i].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax) 
            ims.append([im]) 
        anim = ArtistAnimation(fig, ims, interval=interval)
        if path is not None:
            file_3D_save = os.path.join(path, filename)
            anim.save(file_3D_save + ".gif")
        if show:
            plt.show()


def plot_series(moving, fixed, coreg, path=None, filename='animation', vmin=None, vmax=None, interval=250, show=False):
    titlesize=6
    fig, ax = plt.subplots(figsize=(6, 2), ncols=3, nrows=1)
    ax[0].set_title('model fit', fontsize=titlesize)
    ax[1].set_title('data', fontsize=titlesize)
    ax[2].set_title('coregistered', fontsize=titlesize)
    for i in range(3):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
    ax[0].imshow(fixed[:,:,0].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
    ax[1].imshow(moving[:,:,0].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax) 
    ax[2].imshow(coreg[:,:,0].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax) 
    ims = []
    for i in range(fixed.shape[-1]):
        im0 = ax[0].imshow(fixed[:,:,i].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
        im1 = ax[1].imshow(moving[:,:,i].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax) 
        im2 = ax[2].imshow(coreg[:,:,i].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)  
        ims.append([im0, im1, im2]) 
    anim = ArtistAnimation(fig, ims, interval=interval, repeat_delay=interval)
    if path is not None:
        file_3D_save = os.path.join(path, filename)
        anim.save(file_3D_save + ".gif")
    if show:
        plt.show()   
    else:
        plt.close() 


def plot_coreg(moving, fixed, coreg, defo, dmax=2.0, vmax=10000):

    titlesize = 6
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6), ncols=4, nrows=3)
    for i in range(3):
        for j in range(4):
            ax[i,j].set_yticklabels([])
            ax[i,j].set_xticklabels([])

    ax[0,0].imshow(fixed.T, cmap='gray', vmin=0, vmax=vmax)
    ax[0,0].set_title('fixed image', fontsize=titlesize)
    ax[1,0].imshow(moving.T, cmap='gray', vmin=0, vmax=vmax)
    ax[1,0].set_title('moving image', fontsize=titlesize)
    ax[2,0].imshow((fixed - moving).T, cmap='gray', vmin=-vmax/4, vmax=vmax/4)
    ax[2,0].set_title('fixed - moving', fontsize=titlesize)

    ax[0,1].imshow(fixed.T, cmap='gray', vmin=0, vmax=vmax)
    ax[0,1].set_title('fixed image', fontsize=titlesize)
    ax[1,1].imshow(coreg.T, cmap='gray', vmin=0, vmax=vmax)
    ax[1,1].set_title('deformed', fontsize=titlesize)
    ax[2,1].imshow((fixed - coreg).T, cmap='gray', vmin=-vmax/4, vmax=vmax/4)
    ax[2,1].set_title('fixed - deformed', fontsize=titlesize)

    ax[0,2].imshow(moving.T, cmap='gray', vmin=0, vmax=vmax)
    ax[0,2].set_title('moving image', fontsize=titlesize)
    ax[1,2].imshow(coreg.T, cmap='gray', vmin=0, vmax=vmax)
    ax[1,2].set_title('deformed', fontsize=titlesize)
    ax[2,2].imshow((moving - coreg).T, cmap='gray', vmin=-vmax/4, vmax=vmax/4)
    ax[2,2].set_title('moving - deformed', fontsize=titlesize)

    
    ax[0,3].imshow(defo[:,:,0].T, cmap='gray', vmin=-dmax, vmax=dmax)
    ax[0,3].set_title('deformation (vertical)', fontsize=titlesize)
    ax[1,3].imshow(defo[:,:,1].T, cmap='gray', vmin=-dmax, vmax=dmax)
    ax[1,3].set_title('deformation (horizontal)', fontsize=titlesize)
    ax[2,3].imshow(np.sqrt(defo[:,:,0]**2+defo[:,:,1]**2).T, cmap='gray', vmin=0, vmax=dmax)
    ax[2,3].set_title('deformation (magnitude)', fontsize=titlesize)

    plt.show()



def params(array, path, filename, bounds=[-np.inf, np.inf]):

    file = os.path.join(path, filename + '.png')
    array[np.isnan(array)] = 0 
    array[np.isinf(array)] = 0
    array = np.clip(array, bounds[0], bounds[1])
    shape_arr = np.shape(array)

    if len(shape_arr) == 2: #2D data save
        plt.imshow((array).T)
        plt.clim(np.amin(array), np.amax(array))
        cBar = plt.colorbar()
        cBar.minorticks_on()
        plt.savefig(fname=file)
        plt.close()
    else: #3D data save
        file_3D = os.path.join(path, filename)
        for i in range(shape_arr[2]):
            plt.imshow((array[:,:,i]).T)
            plt.clim(np.amin(array), np.amax(array))
            cBar = plt.colorbar()
            cBar.minorticks_on()
            plt.savefig(fname=[file_3D + '_' + str(i) + ".png"])
            plt.close()