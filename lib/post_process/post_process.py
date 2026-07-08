######################################################
# DDLES: Data-driven model for Large Eddy Simulation  #
# Georgios Momferatos, 2022-2023                      #
# g.momferatos@ipta.demokritos.gr                     #
#######################################################

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split

from lib.datasets.TurbDataset import TurbDataset

def plot_results(args, model, train_losses, test_losses,
                 dataset, dataloader, scaler):
    
    """Plot results

    Parameters
    ----------
    model : PyTorch model
       Model to use

    filename : list of strs
       HDF5 filename

    train_losses : list of floats
       Training losses

    test_losses : list of floats
       Test losses

    args : Namespace
       Namespace holding global parameters
    
    dataset : PyTorch dataset
       Dataset

    Returns
    -------
    None

    """
    
    # load colormap for visualization
    script_path = os.path.join(*(os.path.split(
        os.path.realpath(__file__))[:-1]))
    ncmap_blueblack = np.load(os.path.join(script_path,
                                       'blue-black_cmap.npy')) / 255.
    cmap = ListedColormap(ncmap_blueblack)

#    X_norm = [dataset.X_mean, dataset.X_std]
#    y_norm = [dataset.y_mean, dataset.y_std]


    with torch.no_grad():
        y = next(iter(dataloader)).to(args.device)
        # y = y.unsqueeze(0).unsqueeze(0)
        y = dataset.truncate(y)
        X = dataset.LES_filter(y)

        #!!!!

        X_sc, _ = scaler.transform(X, y, action='scale')
        
        y_pred = model(X_sc)
        
        
        div = dataset.divergence(y_pred).item()
        print(f'maxdiv: {div}')
        
        _, y_pred = scaler.transform(X, y_pred, action='unscale')

        filtered_X = dataset.LES_filter(X)

        filtered_y = dataset.LES_filter(y)

        filtered_y_pred = dataset.LES_filter(y_pred)


        aux = y[-1].to('cpu')
        ky, sy = spectrum(aux, args)
        aux = y_pred[-1].to('cpu')
        ky_pred, sy_pred = spectrum(aux, args)
        aux = X[-1].to('cpu')
        kX, sX = spectrum(aux, args)

        fig = plt.figure(figsize=(12.5, 7.5))
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Test loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE loss')
        plt.title('Training/Test losses')
        plt.legend(loc='best')
        plt.savefig(os.path.join(args.out, 'losses.png'))

        fig = plt.figure(figsize=(12.5, 7.5))
        plt.loglog(kX, sX, color='blue', label='Feature $X$')
        plt.loglog(ky, sy, color='green', label='Target $y$')
        plt.loglog(ky_pred, sy_pred, color='black',
                         label='Prediction $y_p$')
        k_LES_cutoff = args.alpha * np.max(kX)
        plt.axvline(x=k_LES_cutoff, color='red',
                          linestyle='--', label=r'LES filter cutoff $2 \pi / \Delta$')
        k_DNS_cutoff = np.sqrt(2.0) / 3.0 * np.max(kX)
        plt.axvline(x=k_DNS_cutoff, color='orange',
                        linestyle='--', label=r'DNS resolution cutoff $\simeq 2 \pi / \eta$')
        plt.ylabel('$E(k)$')
        plt.xlabel('Wavenumber $k$')
        plt.legend(loc='best')
        plt.title('Energy spectra')
        plt.savefig(os.path.join(args.out, 'spectra.png'))
        
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        
        
        aux = batch_to_numpy(X, dataset)
        axs[0, 0].imshow(aux[-1], cmap=cmap)
        title = 'Feature X'
        axs[0, 0].set_title(title)

        aux = batch_to_numpy(y, dataset)
        axs[0, 1].imshow(aux[-1], cmap=cmap)
        title = 'Target y'
        axs[0, 1].set_title(title)

        aux = batch_to_numpy(y_pred, dataset)
        axs[0, 2].imshow(aux[-1], cmap=cmap)
        axs[0, 2].set_title('Prediction $y_p$')


        axs[1, 0].plot(train_losses, label='Training loss')
        axs[1, 0].plot(test_losses, label='Test loss')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('MSE loss')
        axs[1, 0].set_title('Training/Test losses')
        axs[1, 0].legend(loc='best')

        aux = batch_to_numpy(filtered_y, dataset)
        axs[1, 1].imshow(aux[-1], cmap=cmap)
        title = r'Target large scales $\overline{y} = X$'
        axs[1, 1].set_title(title)

        aux = batch_to_numpy(filtered_y_pred, dataset)
        axs[1, 2].imshow(aux[-1], cmap=cmap)
        axs[1, 2].set_title(r'Predicted large scales $\overline{y}_p$')

        axs[2, 0].loglog(kX, sX, color='blue', label='Feature $X$')
        axs[2, 0].loglog(ky, sy, color='green', label='Target $y$')
        axs[2, 0].loglog(ky_pred, sy_pred, color='black',
                         label='Prediction $y_p$')
        k_LES_cutoff = args.alpha * np.max(kX)
        axs[2, 0].axvline(x=k_LES_cutoff, color='red',
                          linestyle='--', label='LES filter cutoff')
        k_DNS_cutoff = np.sqrt(2.0) / 3.0 * np.max(kX)
        axs[2, 0].axvline(x=k_DNS_cutoff, color='orange',
                        linestyle='--', label='DNS resolution cutoff')
        axs[2, 0].set_ylabel('$E(k)$')
        axs[2, 0].set_xlabel('Wavenumber $k$')
        axs[2, 0].legend(loc='best')
        axs[2, 0].set_title('Energy spectra')

        aux = y - filtered_y
        aux = batch_to_numpy(aux, dataset)
        axs[2, 1].imshow(aux[-1], cmap=cmap)
        title = r'Target small scales $y - \overline{y}$'
        axs[2, 1].set_title(title)

        aux = y_pred - filtered_y_pred
        aux = batch_to_numpy(aux, dataset)
        axs[2, 2].imshow(aux[-1], cmap=cmap)
        axs[2, 2].set_title(r'Predicted small scales $y_p - \overline{y}_p$')

        for iax, ax in enumerate(axs.ravel()):
            if iax != 3 and iax != 6:
                ax.tick_params(left=False, bottom=False,
                               labelleft=False, labelbottom=False)
    plt.tight_layout()
    
    #plt.show()
    plt.savefig(os.path.join(args.out, f'{args.model}.png'))

    if args.rank == 0:
        h5_filename = f'{args.model}.h5'
        filename = os.path.join(args.out, h5_filename)
        with h5py.File(filename, 'w') as h5file:
            aux = batch_to_numpy(y, dataset)
            h5file['y'] = aux

            aux = batch_to_numpy(X, dataset)
            h5file['X'] = aux

            aux = batch_to_numpy(y_pred, dataset)
            h5file['y_pred'] = aux

            aux = batch_to_numpy(filtered_y, dataset)
            h5file['ls_y'] = aux

            aux = batch_to_numpy(filtered_X, dataset)
            h5file['ls_X'] = aux

            aux = batch_to_numpy(filtered_y_pred, dataset)
            h5file['ls_y_pred'] = aux

            aux = batch_to_numpy(y - filtered_y, dataset)
            h5file['ss_y'] = aux

            aux = batch_to_numpy(X - filtered_X, dataset)
            h5file['ss_X'] = aux

            aux = batch_to_numpy(y_pred - filtered_y_pred, dataset)
            h5file['ss_y_pred'] = aux

        xmf_filename = ('.'.join(filename.split('.')[:-1] +
                                 ['xmf']))
        write_xdmf_file(h5_filename,
                        xmf_filename, args)

    # no barrier here: only rank 0 plots, and end-of-run synchronization is
    # cleanup_distributed()'s barrier in the caller

    return


def batch_to_numpy(X, dataset):

    X = dataset.vorticity(X)
    X = torch.linalg.vector_norm(X, dim=1)
    
    #X = dataset.subgrid_scale_tensor(X)
    #X = torch.linalg.matrix_norm(X, dim=(1, 2))

    X = np.array(X[-1].to('cpu'))

    return X

def write_xdmf_file(h5_filename, xmf_filename, args):
    """Writes Xdmf file for visualzation of the corresponding HDF5 file with 
       Paraview

    Parameters
    ----------
    num_file : int
       Number of file 

    xmf_filename : str
       Filename of Xdmf file

    args : Namespace
       Namespace holding global parameters

    """

    if args.dimensions == 2:
        num_el_str = ' 1    {}    {}"'.format(*(2 * [args.n]))
        dimensions_str = (' Dimensions ' + 
        '="    1    {}    {}" '.format(*(2 * [args.n])))

    else:
        num_el_str = ' {}    {}    {}"'.format(*(3 * [args.n]))
        dimensions_str = (' Dimensions ' + 
        '="    {}    {}    {}" '.format(*(3 * [args.n])))

    with open(xmf_filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" '
                ' Version="3.0">\n')
        f.write('  <Domain>\n')
        f.write('    <Grid Name="Grid">\n')
        f.write('      <Geometry Origin="" Type="ORIGIN_DXDYDZ">\n')
        f.write('        <DataItem DataType="Float" Dimensions="3"'
                ' Format="XML"\n')
        f.write('	Precision="8">0 0 0</DataItem>\n')
        f.write('        <DataItem DataType="Float" Dimensions="3" '
                ' Format="XML"\n')
        f.write('	Precision="8">1 1 1</DataItem>\n')
        f.write('      </Geometry>\n')
        f.write(('      <Topology NumberOfElements="' + num_el_str
                 + ' Type="3DCoRectMesh"/>\n'))

        f.write('      <Attribute Center="Node" Name="X"'
                ' Type="Scalar">\n')
        f.write(('        <DataItem DataType="Float" Precision="     4"' 
            + dimensions_str))
        f.write(f'  Format="HDF">{h5_filename}:/X'
                '</DataItem>\n')
        f.write('      </Attribute>\n')

        f.write('      <Attribute Center="Node" Name="y"'
                ' Type="Scalar">\n')
        f.write(('        <DataItem DataType="Float" Precision="     4"' 
            + dimensions_str))
        f.write(f'  Format="HDF">{h5_filename}:/y</DataItem>\n')
        f.write('      </Attribute>\n')

        f.write('      <Attribute Center="Node" Name="y_pred"'
                ' Type="Scalar">\n')
        f.write(('        <DataItem DataType="Float" Precision="     4"' 
            + dimensions_str))
        f.write(f'  Format="HDF">{h5_filename}:/y_pred</DataItem>\n')
        f.write('      </Attribute>\n')

        #

        f.write('      <Attribute Center="Node" Name="ls_X"'
                ' Type="Scalar">\n')
        f.write(('        <DataItem DataType="Float" Precision="     4"' 
            + dimensions_str))
        f.write(f'  Format="HDF">{h5_filename}:/ls_X'
                '</DataItem>\n')
        f.write('      </Attribute>\n')

        f.write('      <Attribute Center="Node" Name="ls_y"'
                ' Type="Scalar">\n')
        f.write(('        <DataItem DataType="Float" Precision="     4"' 
            + dimensions_str))
        f.write(f'  Format="HDF">{h5_filename}:/ls_y</DataItem>\n')
        f.write('      </Attribute>\n')

        f.write('      <Attribute Center="Node" Name="ls_y_pred"'
                ' Type="Scalar">\n')
        f.write(('        <DataItem DataType="Float" Precision="     4"' 
            + dimensions_str))
        f.write(f'  Format="HDF">{h5_filename}:/ls_y_pred</DataItem>\n')
        f.write('      </Attribute>\n')

        #

        f.write('      <Attribute Center="Node" Name="ss_X"'
                ' Type="Scalar">\n')
        f.write(('        <DataItem DataType="Float" Precision="     4"' 
            + dimensions_str))
        f.write(f'  Format="HDF">{h5_filename}:/ss_X'
                '</DataItem>\n')
        f.write('      </Attribute>\n')

        f.write('      <Attribute Center="Node" Name="ss_y"'
                ' Type="Scalar">\n')
        f.write(('        <DataItem DataType="Float" Precision="     4"' 
            + dimensions_str))
        f.write(f'  Format="HDF">{h5_filename}:/ss_y</DataItem>\n')
        f.write('      </Attribute>\n')

        f.write('      <Attribute Center="Node" Name="ss_y_pred"'
                ' Type="Scalar">\n')
        f.write(('        <DataItem DataType="Float" Precision="     4"' 
                 + dimensions_str))
        f.write(f'  Format="HDF">{h5_filename}:/ss_y_pred</DataItem>\n')
        f.write('      </Attribute>\n')



        f.write('    </Grid>\n')
        f.write('  </Domain>\n')
        f.write('</Xdmf>\n')

def spectrum(X, args):
    """Returns the energy spectrum of a single sample

    Parameters
    ----------
    X : 3D/4D tensor of floats
       Input field: leading channel dimension(s), spatial dimensions last

    """
    n = X.shape[-2]
    dims = (-2, -1) if args.dimensions == 2 else (-3, -2, -1)
    # forward real-to-half-complex FFT
    fX = torch.fft.rfftn(X, dim=dims, norm='ortho')

    wvs = torch.fft.fftfreq(n) # wavenumbers
    rwvs = torch.fft.rfftfreq(n) # real-to-half-complex wavenumbers
    # wavevector magnitude at every retained mode
    if args.dimensions == 2:
        wvms = torch.sqrt(wvs.view(-1, 1) ** 2 +
                          rwvs.view(1, -1) ** 2)
    else:
        wvms = torch.sqrt(wvs.view(-1, 1, 1) ** 2 +
                          wvs.view(1, -1, 1) ** 2 +
                          rwvs.view(1, 1, -1) ** 2)
    wvms = wvms.to(fX.device)

    # bin centers of the spectrum
    wvs_spec = torch.linspace(0.0, torch.max(wvms), n).to(fX.device)

    # energy at every mode, summed over the leading (channel) dims
    nrg = (torch.abs(fX) ** 2).reshape(-1, *wvms.shape).sum(dim=0)

    # nearest-center binning, as the previous argmin loops did: bucketize
    # against the midpoints between centers, then accumulate every mode in
    # one index_add_ (duplicate bins add up, unlike buffered `spec[idx] +=`)
    midpoints = 0.5 * (wvs_spec[1:] + wvs_spec[:-1])
    idx = torch.bucketize(wvms.reshape(-1), midpoints)
    spec = torch.zeros_like(wvs_spec)
    spec.index_add_(0, idx, nrg.reshape(-1))

    wvs_spec = np.array(wvs_spec.to('cpu'))
    spec = np.array(spec.to('cpu'))

    return wvs_spec, spec

def plot_FNet(model, args):
    plt.figure(figsize=(15, 10))
    plt.xscale('log')
    for param_tensor in model.state_dict():
        if 'alpha' in param_tensor:
            plt.plot(np.array(
                model.state_dict()[param_tensor].to('cpu')),
                     label=f'{param_tensor}')

    plt.legend(loc='best')
    
    plt.savefig(os.path.join(args.out, 'alphas.png'))
    return

def plot_histograms(dataloader, model, dataset, scaler, args):

    fig = plt.figure(figsize=(10.0, 7.5))
    plt.yscale('log')
    nbatches = len(dataloader)

    with torch.no_grad():
        y = next(iter(dataloader)).to(args.device)
        # y = y.unsqueeze(0).unsqueeze(0)
        y = dataset.truncate(y)
        
        X = dataset.LES_filter(y)
        
        X, _ = scaler.transform(X, y, action='scale')
        
        y_pred = model(X)

        _, y_pred = scaler.transform(X, y_pred, action='unscale')
        
        maxstep = int(np.log2(args.n))

        hist_xs = []
        hist_ys = []
        hist_xs_pred = []
        hist_ys_pred = []
        hist_wvs = []
        hist_steps = []
        delta_x = 2.0 * np.pi 

        aux = y[-1].to('cpu')
        ky, sy = spectrum(aux, args)
        aux = y_pred[-1].to('cpu')
        ky_pred, sy_pred = spectrum(aux, args)
        aux = X[-1].to('cpu')
        kX, sX = spectrum(aux, args)
        kmax = np.max(kX)
        eta = 2.0 * np.pi / kmax
        delta_x = 2.0 * eta
        
        colors = ('red', 'green', 'blue', 'cyan', 'magenta', 'orange', 'purple', 'pink', 'yellow', 'gray', 'gold')
        for istep in range(0, maxstep):
            step = 2 ** istep
            color = colors[istep] 
            shifts = 3 * [step] + 3 * [-step]
            dims = 2 * [-3, -2, -1]
            for yy, var, style in zip((y, y_pred), \
                                       ('Target', 'Prediction'), ('-', '--')):
                data = []
                for shift, dim in zip(shifts, dims):
                    yyrolled = torch.roll(yy, shifts=shift, dims=dim)
                    diff = yy - yyrolled
                    vec = torch.zeros_like(diff)
                    # sign of the separation direction, not step: step is
                    # always positive, so np.sign(step) gave the negative
                    # shifts a +1 longitudinal direction, flipping their sign
                    # and symmetrizing the increment PDF (killing its skewness)
                    vec[:, dim, :, :] = np.sign(shift)
                    incr = torch.linalg.vecdot(vec, diff, dim=1)
                    incr = incr.flatten()
                    data.append(incr)
                data = torch.cat(data)
                std = data.std().to('cpu').item()
                data = data.to('cpu')
                yp, xp = torch.histogram(data, bins=args.n, density=True)
                xp = xp.to('cpu').numpy()
                xp = 0.5 *(xp[1:] + xp[:-1])
                yp = yp.to('cpu').numpy()
                
                if var == 'Target':
                    label = rf'$r = {step}~\Delta x$'
                else:
                    label = None
                   
                plt.plot(xp / std, yp * std ,
                         style, color=color, label=label)

                if var == 'Target':
                    hist_xs.append(xp / std)
                    hist_ys.append(yp * std)
                else:
                    hist_xs_pred.append(xp / std)
                    hist_ys_pred.append(yp * std)
            hist_wvs.append(2.0 * np.pi / (step * delta_x))
            hist_steps.append(step)
                
    plt.title('Longitudinal velocity increment PDFs')
    plt.ylabel(r'$\sigma_{\delta u_L} P(\delta u_L)$')
    plt.xlabel(r'$\delta u_L / \sigma_{\delta u_L}$')
    xp = np.linspace(-6.0, 6.0, args.n)
    gauss_x = xp
    gauss_y = 1. / np.sqrt(2 * np.pi) * np.exp(-0.5 * xp ** 2)
    plt.plot(gauss_x, gauss_y, '.', color='black', label='Gaussian')
    plt.legend(loc='best')
    plt.savefig(os.path.join(args.out, 'hist_vel_incrs.png'))

    for iplot, (hist_x,
                hist_y, hist_x_pred, hist_y_pred, hist_wv,
                hist_step) in enumerate(zip(hist_xs, hist_ys,
                                            hist_xs_pred,
                                            hist_ys_pred, hist_wvs,
                                            hist_steps)):
        fig, axs = plt.subplots(1, 2, figsize=(20.0, 7.5))
        axs[0].loglog(kX, sX, color='blue', label='Feature $X$')
        axs[0].loglog(ky, sy, color='green', label='Target $y$')
        axs[0].loglog(ky_pred, sy_pred, color='black',
                         label='Prediction $y_p$')
        k_LES_cutoff = args.alpha * np.max(kX)
        axs[0].axvline(x=k_LES_cutoff, color='red',
                          linestyle='--', label=r'LES filter cutoff $2 \pi / \Delta$')
        k_DNS_cutoff = np.sqrt(2.0) / 3.0 * np.max(kX)
        axs[0].axvline(x=k_DNS_cutoff, color='orange',
                        linestyle='--', label=r'DNS resolution cutoff $\simeq 2 \pi / \eta$')
        axs[0].axvline(x=hist_wv, color='purple',
                          linestyle='--', label=r'increment wavenumber $ 2 \pi / r$')
        axs[0].set_ylabel('$E(k)$')
        axs[0].set_xlabel('Wavenumber $k$')
        axs[0].legend(loc='best')
        axs[0].set_title('Energy spectra')

        axs[1].set_yscale('log')
        axs[1].plot(hist_x, hist_y, color='purple', label = rf'$P(\delta u_L(r)), r = {hist_step}~\Delta x$')
        axs[1].plot(hist_x_pred, hist_y_pred, '--', color='purple', label = 'Prediction')
        axs[1].plot(gauss_x, gauss_y, '.', color='black', label = 'Gaussian distribution')
        axs[1].set_title(rf'Longitudinal velocity increment PDFs, $r = {hist_step}~\Delta x$')
        axs[1].set_ylabel(r'$\sigma_{\delta u_L} P(\delta u_L)$')
        axs[1].set_xlabel(r'$\delta u_L / \sigma_{\delta u_L}$')
        axs[1].legend(loc='best')
        plt.savefig(os.path.join(args.out, f'hist_vel_incr_{iplot:03d}.png'))
    
    fig = plt.figure(figsize=(10.0, 7.5))
    plt.yscale('log')
    for data, label, style in zip((y, y_pred), \
                                   ('Target', 'Prediction'), ('-', '--')):
        lgrads = dataset.longitudinal_gradients(data)
        std = lgrads.std().to('cpu').item()
        lgrads = lgrads.to('cpu')
        yp, xp = torch.histogram(lgrads, bins=args.n, density=True)
        xp = xp.to('cpu').numpy()
        xp = 0.5 *(xp[1:] + xp[:-1])
        yp = yp.to('cpu').numpy()
        plt.plot(xp / std, yp * std , style, color='black', label=label)
        plt.title('Velocity gradient PDFs')
        plt.ylabel(r'$\sigma P(\partial u / \partial x)$')
        plt.xlabel(r'$(\partial u / \partial x)/\sigma$')
    
    xp = np.linspace(-6.0, 6.0, args.n)
    plt.plot(xp, 1. / np.sqrt(2 * np.pi) * np.exp(-0.5 * xp ** 2),
                '.', color='black', label='Gaussian')
    plt.legend(loc='best')
    
    plt.savefig(os.path.join(args.out, 'hist_vel_grad.png'))
    
    return
