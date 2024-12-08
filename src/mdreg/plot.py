import os
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import numpy as np
import math
import inspect


def animation(array, path=None, filename='animation', vmin=None, vmax=None, 
              slice=None, title = '', interval=250, show=False):

    """
    Produce an animation of a 3D image.

    Parameters
    ----------
    array : numpy.array
        The 3D image to animate.
    path : str, optional
        The path to save the animation. The default is None.
    filename : str, optional
        The filename of the animation. The default is 'animation'.
    vmin : float, optional
        The minimum value for the colormap. The default is None.
    vmax : float, optional
        The maximum value for the colormap. The default is None.
    slice : int, optional
        The slice to plot for 3D data. The default is None which plots all slices.
    title : str, optional
        The title of the animation to be rendered on the figure.
        The default is ''.
    interval : int, optional
        The interval between frames. The default is 250ms.
    show : bool, optional
        Whether to display the animation. The default is False.

    """

    array[np.isnan(array)] = 0
    shape = np.shape(array)
    titlesize = 10

    if array.ndim == 4 and slice is None : ##save 3D data

        # Determine the grid size for the panels
        num_slices = array.shape[2]
        grid_size = math.ceil(math.sqrt(num_slices))

        fig_3d, axes1 = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
        fig_3d.subplots_adjust(wspace=0.5, hspace=0.01)

        fig_3d.suptitle('{} \n \n'.format(title), fontsize=titlesize+2)
        plt.tight_layout()

        for i in range(grid_size * grid_size):
                row = i // grid_size
                col = i % grid_size
                if i < num_slices:
                    axes1[row, col].imshow(array[:, :, i, 0].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
                    axes1[row, col].set_title('Slice {}'.format(i+1), fontsize=titlesize)
                else:
                    axes1[row, col].axis('off')  # Turn off unused subplots
                axes1[row, col].set_xticks([])  # Remove x-axis ticks
                axes1[row, col].set_yticks([])

        images = []
        for j in range(array.shape[-1]):
            ims = []
            for i in range(grid_size * grid_size):
                row = i // grid_size
                col = i % grid_size
                if i < num_slices:
                    im = axes1[row, col].imshow(array[:, :, i, j].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
                    ims.append(im)
            images.append(ims,)

        anim = ArtistAnimation(fig_3d, images, interval=interval, repeat_delay=interval)
        if path is not None:
            file_3D_save = os.path.join(path, filename)
            anim.save(file_3D_save + "_"  + ".gif")
        if show:
            plt.show()
            return anim
        else:
            plt.close()
            return
    
    elif array.ndim-1 == 3 and slice is not None: # save 3D data
        array = array[:,:,slice,:]

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
            return anim
        else:
            plt.close()
            return


def plot_series(moving, fixed, coreg, path=None, filename='animation', 
                vmin=None, vmax=None, slice=None, interval=250, show=False):

    """
    Produce an animation of the original, fitted and coregistered images.

    Parameters
    ----------
    moving : numpy.array
        The moving image.
    fixed : numpy.array
        The fixed/fitted image.
    coreg : numpy.array
        The coregistered image.
    path : str, optional
        The path to save the animation. The default is None.
    filename : str, optional
        The filename of the animation. The default is 'animation'.
    vmin : float, optional
        The minimum value for the colormap. The default is None.
    vmax : float, optional
        The maximum value for the colormap. The default is None.
    slice : int, optional
        The slice to plot. The default is None, which plots the central slice.
    interval : int, optional
        The interval between frames. The default is 250ms.
    show : bool, optional
        Whether to display the animation. The default is False.

    """

    titlesize = 6

    if (moving.ndim == fixed.ndim == coreg.ndim == 4) and (slice is None):

        # Determine the grid size for the panels
        num_slices = moving.shape[2]
        grid_size = math.ceil(math.sqrt(num_slices))
        titles = ['Original Data', 'Model Fit', 'Coregistered']
        anims = []

        for data in (moving, fixed, coreg):
            fig_3d, axes1 = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
            fig_3d.subplots_adjust(wspace=0.5, hspace=0.01)
            data_name = titles[[np.array_equal(data, moving), np.array_equal(data, fixed), np.array_equal(data, coreg)].index(True)]

            fig_3d.suptitle('Series Type: {} \n \n'.format(data_name), fontsize=titlesize+2)
            plt.tight_layout()

            for i in range(grid_size * grid_size):
                    row = i // grid_size
                    col = i % grid_size
                    if i < num_slices:
                        axes1[row, col].imshow(data[:, :, i, 0].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
                        axes1[row, col].set_title('Slice {}'.format(i+1), fontsize=titlesize)
                    else:
                        axes1[row, col].axis('off')  # Turn off unused subplots
                    axes1[row, col].set_xticks([])  # Remove x-axis ticks
                    axes1[row, col].set_yticks([])

            images = []
            for j in range(data.shape[-1]):
                ims = []
                for i in range(grid_size * grid_size):
                    row = i // grid_size
                    col = i % grid_size
                    if i < num_slices:
                        im = axes1[row, col].imshow(data[:, :, i, j].T, cmap='gray', animated=True, vmin=vmin, vmax=vmax)
                        ims.append(im)
                images.append(ims,)

            anim = ArtistAnimation(fig_3d, images, interval=interval, repeat_delay=interval)
            if path is not None:
                file_3D_save = os.path.join(path, filename)
                data_type_name = _get_var_name(data)
                anim.save(file_3D_save + "_" + data_type_name + ".gif")
            if show:
                plt.show()
                anims.append(anim)
            else:
                plt.close()
        if show:
            return anims
        else:
            return

    elif (moving.ndim == fixed.ndim == coreg.ndim == 4) and (slice is not None):
            fixed = fixed[:,:,slice,:]
            moving = moving[:,:,slice,:]
            coreg = coreg[:,:,slice,:]

    elif not (moving.ndim == fixed.ndim == coreg.ndim):
        raise ValueError('Dimension mismatch in arrays provided. Please '
                         'ensure the three arrays have the same dimensions')

    fig, ax = plt.subplots(figsize=(6, 2), ncols=3, nrows=1)
    if slice is not None:
        fig.suptitle('Slice {} \n \n'.format(slice), fontsize=titlesize+2)
    ax[0].set_title('Model fit', fontsize=titlesize+2)
    ax[1].set_title('Original Data', fontsize=titlesize+2)
    ax[2].set_title('Coregistered', fontsize=titlesize+2)
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
        return anim 
    else:
        plt.close()
        return



def _plot_coreg(moving, fixed, coreg, defo, dmax=2.0, vmax=10000):

    """
    Plot the moving, fixed and coregistered images, and the deformation field.

    Parameters
    ----------
    moving : numpy.array
        The moving image.
    fixed : numpy.array
        The fixed/fitted image.
    coreg : numpy.array
        The coregistered image.
    defo : numpy.array
        The deformation field.
    dmax : float, optional
        The maximum value for the deformation field. The default is 2.0.
    vmax : float, optional
        The maximum value for the colormap. The default is 10000.
    
    Returns
    -------
    None, shows the plot.    

    """

    if (len(np.shape(moving)[:-1]) == 3):
        raise ValueError("Plotting not compatible for 3D dynamic data.")

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

    return 


def _plot_params(array, path, filename, bounds=[-np.inf, np.inf]):

    """
    Plot and save the parameters of the model.

    Parameters
    ----------
    array : numpy.array
        The array of parameters to plot.
    path : str
        The path to save the plot.
    filename : str
        The filename of the plot.
    bounds : list, optional
        The bounds of the colormap. The default is [-np.inf, np.inf].
    
    Returns
    -------
    None, saves the plot.

    """

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

    return

def _get_var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    for var_name, var_val in callers_local_vars:
        if var_val is var:
            return var_name
    return None
