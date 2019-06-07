"""Plot results"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmc_robot import ExperimentLogger
from save_figures import save_figures
from parse_args import save_plots


def plot_positions(times, link_data):
    """Plot positions"""
    
#    plt.title('Body joint angles', fontsize=20)
#    for i, data in enumerate(link_data.T):
#        
#        plt.plot(times, data+2*i, label='joint '+str(i+1))
#        
#        plt.xlabel("Time [s]", fontsize=18)
#        #plt.ylabel("Distance [m]")
#        plt.grid(True)
#    
#    plt.legend(prop={'size': 12})
    
    plt.title('Limb joint angles', fontsize=20)
    plt.plot(times, link_data)
    plt.xlabel("Time [s]",  fontsize=18)
    plt.grid(True)


def plot_trajectory(link_data):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 2])
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.grid(True)


def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear'  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], "r.")
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation="none",
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])


def main(plot=True):
    """Main"""
    # Load data
    with np.load('logs/simulation9g.npz') as data:
        timestep = float(data["timestep"])
        #amplitude = data["amplitudes"]
        #phase_lag = data["phase_lag"]
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]
    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
    print(type(joints_data))
    # Plot data
    #plt.figure("Positions")
    
    
#    data = np.load("logs/simulation9d2.npz")
#    link_data = data["links"][:, 0, :]
#    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)
#    plot_positions(times, link_data)
    
#    plt.figure("9b")
#    phi = np.array([ 2*np.pi/(3*10), 2*np.pi/(2*10),2*np.pi/10,2*2*np.pi/10, 3*2*np.pi/100])
#    amp = np.linspace(0.1,5.0,25)
#    energy = np.zeros((25,5))
#    for ind, angle in enumerate(phi):
#        for j in range(0, len(amp)):
#            data = np.load("logs/simulation9b_phi{}_amp{}.npz".format(ind,j))
#            velocity = np.diff(data["joints"][:,:,0], axis = 0)/timestep
#            velocity = np.insert(velocity,0,0,axis = 0)
#            torque = data["joints"][:,:,2]
#            force = velocity*torque
#            totalforce = np.zeros(len(data["joints"][:,1,2]))
#            for a in range (0,len(data["joints"][:,1,2])):
#                totalforce[a]= sum(force[a,:])
#            energy[j,ind] = (np.trapz(abs(totalforce),dx = timestep))
#
#    #outfile.seek(0) #for when you want to open it again
#    plt.imshow(energy.T, extent = [2*3.14/(3*10),6*3.14/10,0,5])
#    plt.xlabel('phase lag')
#    plt.ylabel('amplitude factor')
#    plt.title('Energy estimation')
#    plt.colorbar()
#    plt.show()
    
#    plt.figure("9c")
#    head = np.linspace(0.1,1.5,15)
#    tail = np.linspace(0.1,1.5,15)
#    energy = np.zeros((len(head), len(tail)))
#    speed = np.zeros((len(head), len(tail)))
#    for i in range(0 ,len(head)):
#        for j in range(0,len(tail)):
#            data = np.load("logs/simulation9c_more_head{}_tail{}.npz".format(head[i],tail[j]))
#            velocity = np.diff(data["joints"][:,:,0], axis = 0)/timestep
#            velocity = np.insert(velocity,0,0,axis = 0)
#            velo = np.diff(data["links"][:,:,0], axis = 0)/timestep
#            velo = np.insert(velocity,0,0,axis = 0)
#            torque = data["joints"][:,:,2]
#            force = velocity*torque
#            totalforce = np.zeros(len(data["joints"][:,1,2]))
#            for a in range (0,len(data["joints"][:,1,2])):
#                totalforce[a]= sum(force[a,:])
#            energy[i,j] = (np.trapz(abs(totalforce),dx = timestep))
#            speed[i,j] = np.mean(abs(velo))
#            
#
#    #outfile.seek(0) #for when you want to open it again
#    plt.imshow(energy, extent = [0.1,1.5,0.1,1.5])
#    plt.xlabel('tail factor')
#    plt.ylabel('head factor')
#    plt.title('Energy estimation')
#    plt.colorbar()
#    plt.show()
#    
#    plt.figure("9c velocity")
#    plt.imshow(energy, extent = [0.1,1.5,0.1,1.5])
#    plt.xlabel('tail factor')
#    plt.ylabel('head factor')
#    plt.title('mean velocity estimation')
#    plt.colorbar()
#    plt.show()
    
    
    
    
#    plt.plot(np.linspace(0,2,20), speed)
#    plt.xlabel('amplitude')
#    plt.ylabel('mean velocity')
#    plt.title('Velocity as a function of oscillation amplitude')
#    plt.show()
    
    
    data = np.load("logs/simulation9g.npz")
    angles = data["joints"][:,:,0] #joint position
    x_coord = data["links"][:,:,0]
    
    plt.figure()
    plt.title('Land/water transition: x coordinate')
    plt.plot(times, x_coord)
    plt.xlabel('Times [s]')
    plt.ylabel('x coordinate [m]')
    
    plt.figure()
    plot_positions(times, angles[:,:10])
    
    plt.figure()
    plot_positions(times, angles[:,10:]%(2*np.pi))

    

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

