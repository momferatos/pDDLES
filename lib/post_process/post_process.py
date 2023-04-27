
#######################################################
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
import torch.distributed as dist

from lib.datasets.Datasets import TurbDataset

def plot_results(args, model, train_losses, test_losses,
                 dataset, dataloader):
    
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
        ymean = y.mean()
        ystd = y.std()

        X = dataset.LES_filter(y)
        Xmean = X.mean()
        Xstd = X.std()
        X = (X - Xmean) / Xstd
        
     
        y_pred = model(X)
        y_pred = y_pred * ystd + ymean
        
        div = dataset.divergence(y_pred).item()
        print(f'maxdiv: {div}')

        # X, y = scaler.transform(X, y, direction='backward')
        
        # _, y_pred = scaler.transform(X, y_pred, direction='backward')

        filtered_X = dataset.LES_filter(X)

        filtered_y = dataset.LES_filter(y)

        filtered_y_pred = dataset.LES_filter(y_pred)


        aux = y[-1].to('cpu')
        ky, sy = spectrum(aux, args)
        aux = y_pred[-1].to('cpu')
        ky_pred, sy_pred = spectrum(aux, args)
        aux = X[-1].to('cpu')
        kX, sX = spectrum(aux, args)

        fig, axs = plt.subplots(3, 3, figsize=(15, 10))

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
        title = 'Target large scales $\overline{y} = X$'
        axs[1, 1].set_title(title)

        aux = batch_to_numpy(filtered_y_pred, dataset)
        axs[1, 2].imshow(aux[-1], cmap=cmap)
        axs[1, 2].set_title('Predicted large scales $\overline{y}_p$')

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
        title = 'Target small scales $y - \overline{y}$'
        axs[2, 1].set_title(title)

        aux = y_pred - filtered_y_pred
        aux = batch_to_numpy(aux, dataset)
        axs[2, 2].imshow(aux[-1], cmap=cmap)
        axs[2, 2].set_title('Predicted small scales $y_p - \overline{y}_p$')

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
        
    dist.barrier()
    
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

def turbify_image(model, fname, alpha):
    """Applies a turbulent filter to an image (just for fun...)

    Parameters
    ----------
    model : PyTorch model
       Model to load

    filename : str
       Filename of image file

    alpha : float
       Fraction of the wavenumber range to be retained by the low-pass

    """

    with torch.no_grad():
        image = Image.open(fname)
        dims = image.size
        n = int(np.log2(np.max(dims)))

        image = image.resize((2 ** n, 2 ** n))
        image = np.asarray(image, dtype=np.uint8)
        image = image / 255.
        image = torch.from_numpy(image[:, :, :])
        image = image.float()
        image = image.to(device)
        image_r = image[:, :, 0].unsqueeze(2)
        image_g = image[:, :, 1].unsqueeze(2)
        image_b = image[:, :, 2].unsqueeze(2)
        seq = []
        for image in [image_r, image_g, image_b]:
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0).to(device)
            image_p = model(image)
            image_p = image_p - LES_filter(image_p, alpha)
            image = image_p.squeeze(0).squeeze(0)
            image = image.to('cpu')
            image = np.array(image)
            image = image / np.max(image)
            seq.append(image)
        image = np.stack(seq)
        image = Image.fromarray(
            np.array(255. * image).astype(np.uint8).transpose(1,2,0))
        image = image.resize(dims)
        outpath = os.path.join(args.session_name, 'image.png')
        image = image.save(outpath)

def spectrum(X, args):
        """Returns the mean (across a minibatch) energy spectrum

        Parameters
        ----------
        x : 4D/5D tensor of floats
           Input scalar field

        """
        n = X.shape[-2]
        batch_size = X.shape[0]
        dims = (-2, -1) if args.dimensions == 2 else (-3, -2, -1)
        # forward FFT
        fX = torch.fft.rfftn(X, dim=dims, norm='ortho')
        wvs = torch.fft.fftfreq(n) # wavenumbers
        rwvs = torch.fft.rfftfreq(n) # real-to-half-complex wavenumbes
        # define wavevector magnitudes
        # 2d
        wvs2d = torch.sqrt(wvs.view(1, -1, 1) ** 2 +
                           rwvs.view(1, 1, -1) ** 2)
        wvs2d = wvs2d.repeat(batch_size, 1, 1, 1)
        # 3d
        wvs3d = torch.sqrt(wvs.view(-1, 1, 1) ** 2 +
                           wvs.view(1, -1, 1) ** 2 +
                           rwvs.view(1, 1, -1) ** 2)
        wvs3d = wvs3d.repeat(batch_size, n, 1, 1, 1)
        wvms = wvs2d if args.dimensions == 2 else wvs3d

        wvmax = torch.max(wvms) # maximum wavevector magnitude

        # define x, y variables for the energy spectrum
        wvs_spec = torch.linspace(0.0, wvmax, n)
        wvs_spec = wvs_spec.repeat(batch_size, 1)
                    
        spec = torch.zeros_like(wvs_spec)
        # calculate the spectrum
        if args.dimensions == 2:
            for k in range(len(rwvs)):
                for j in range(len(wvs)):
                    idx = torch.argmin(torch.abs(wvs_spec[:, :] -
                                                 wvms[:, :, j, k]), dim=1)
                    # accumulate energy at position idx
                    spec[:, idx] += torch.abs(fX[:, :, j, k] ** 2)
        else:
            for k in range(len(rwvs)):
                for j in range(len(wvs)):
                    for i in range(len(wvs)):
                        wvm = wvms[:, :, i, j, k]
                        idx = (torch.argmin(torch.abs(wvs_spec[:, :]
                                                      - wvm[:, :]), dim=1))
                        # accumulate energy at position idx
                        device = spec.device
                        idx = idx.to(device)
                        fX = fX.to(device)
                        nrg = torch.abs(fX[:, i, j, k] ** 2)
                        nrg = nrg.sum(dim=0)
                        spec[:, idx] += nrg

        wvs_spec = wvs_spec[-1]
        wvs_spec = np.array(wvs_spec.to('cpu'))
        spec = torch.mean(spec, dim=0).squeeze(0) # take mean across minibatch
        spec = np.array(spec.to('cpu'))


        return wvs_spec, spec

def plot_FourierNet(model, args):
    plt.figure(figsize=(15, 10))
    plt.xscale('log')
    for param_tensor in model.state_dict():
        if 'alpha' in param_tensor:
            plt.plot(np.array(model.state_dict()[param_tensor].to('cpu')), label=f'{param_tensor}')

    plt.legend(loc='best')
    
    plt.savefig(os.path.join(args.out, 'alphas.png'))
    return


def predict(model, prediction_filenames, args, prediction_dataset,
            prediction_dataloader):
    """Perform prediction

    Parameters
    ----------
    model : PyTorch model
       Model to use

    prediction_filenames : list of strs
       List of filenames

    args: Namespace
       Namespace holding global parameters

    prediction_dataset: PyTorch Dataset
       Prediction dataset

    prediciton_dataloader: PyTorch DataLoader
       Prediction dataloader

    Returns
    -------
    None

    """

    model = model.to(args.device)
    
    
    X_norm = [dataset.X_mean, dataset.X_std]
    y_norm = [dataset.y_mean, dataset.y_std]
   
    prediction_dataset = TurbDataset(prediction_filenames, args)
    prediction_loader = DataLoader(prediction_dataset,
                              batch_size=args.batch_size,
                                   num_workers=args.num_workers)
    prediction_dataset.determine_norm_constants(prediction_loader)
    with torch.no_grad():
        for nbatch, y in enumerate(prediction_loader):
            print('Predicting: '
                  f'Minibatch {nbatch+1:04d}/{len(prediction_loader):04d}.') 
            y,num_file,time = y

            y = y.to(args.device)
            X = prediction_dataset.LES_filter(y)

            if args.prediction_mode == 'large_to_small':
                y = y - X

            y_pred = model(X)

            X = prediction_dataset.normalize(X, -1)
            y = prediction_dataset.normalize(y, -1, feature=False)
            y_pred = prediction_dataset.normalize(y_pred, -1, feature=False)

            filtered_X = prediction_dataset.LES_filter(X)

            filtered_y = prediction_dataset.LES_filter(y)

            filtered_y_pred = prediction_dataset.LES_filter(y_pred)
            num_file0 = 0

            for (yy,XX,yy_pred,filtered_yy,filtered_XX,filtered_yy_pred,
            nnum_file,ttime) in zip(y,X,y_pred,filtered_y,filtered_X,
                                 filtered_y_pred,num_file,time):
                filename = os.path.join(f'{args.session_name}',
                                        'prediction.'
                                        f'{num_file0:06d}.h5')

                with h5py.File(filename, 'w') as h5file:
                    aux = np.array(yy.squeeze(0).to('cpu'))
                    h5file['y'] = aux
                    aux = np.array(XX.squeeze(0).to('cpu'))
                    h5file['X'] = aux
                    aux = np.array(yy_pred.squeeze(0).to('cpu'))
                    h5file['y_pred'] = aux
                    aux = np.array(filtered_yy.squeeze(0).to('cpu'))
                    h5file['ls_y'] = aux
                    aux = np.array(filtered_XX.squeeze(0).to('cpu'))
                    h5file['ls_X'] = aux
                    aux = np.array(filtered_yy_pred.squeeze(0).to('cpu'))
                    h5file['ls_y_pred'] = aux
                    aux = yy - filtered_yy
                    aux = np.array(aux.squeeze(0).to('cpu'))
                    h5file['ss_y'] = aux
                    aux = XX - filtered_XX
                    aux = np.array(aux.squeeze(0).to('cpu'))
                    h5file['ss_X'] = aux
                    aux = yy_pred - filtered_yy_pred
                    aux = np.array(aux.squeeze(0).to('cpu'))
                    h5file['ss_y_pred'] = aux
                    h5file['time'] = ttime.item()
                    h5file['num_file'] = nnum_file.item()

                xmf_filename = ('.'.join(filename.split('.')[:-1] +
                                         ['xmf']))
                write_xdmf_file(num_file0,
                                xmf_filename, args)
                num_file0 += 1

    return

def plot_histograms(dataloader, model, dataset, args):

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    nbatches = len(dataloader)

    with torch.no_grad():
        y = next(iter(dataloader)).to(args.device)
        # y = y.unsqueeze(0).unsqueeze(0)
        y = dataset.truncate(y)
        ymean = y.mean()
        ystd = y.std()

        X = dataset.LES_filter(y)
        Xmean = X.mean()
        Xstd = X.std()
        X = (X - Xmean) / Xstd

        y_pred = model(X)
        y_pred = y_pred * ystd + ymean
        maxstep = int(np.log2(args.n))

        colors = ('red', 'green', 'blue', 'cyan', 'magenta', 'orange', 'purple', 'pink', 'yellow', 'gray', 'gold')
        for istep in range(0, maxstep):
            step = 2 ** istep
            color = colors[istep] 
            shifts = 3 * [step] + 3 * [-step]
            dims = 2 * [-3, -2, -1]
            for yy, var, style in zip((y, y_pred), ('Target', 'Prediction'), ('-', '--')):
                data = []
                for shift, dim in zip(shifts, dims):
                    yyrolled = torch.roll(yy, shifts=shift, dims=dim)
                    diff = yy - yyrolled
                    vec = torch.zeros_like(diff)
                    vec[:, dim, :, :] = np.sign(step)
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
                    label = f'{var}: incr= {step}'
                else:
                    label = None
                   
                axs[0].plot(xp / std, yp * std , style, color=color, label=label)
                
    axs[0].set_title('Longitudinal velocity increment PDFs')
    axs[0].set_ylabel('$\sigma_{\delta u^L} P(\delta u^L)$')
    axs[0].set_xlabel('$\delta u_L / \sigma_{\delta u^L}$')
    xp = xp / std
    axs[0].plot(xp, 1. / np.sqrt(2 * np.pi) * np.exp(-0.5 * xp ** 2), color='black', label='Gaussian')
    axs[0].legend(loc='best')

    for data, label, style in zip((y, y_pred), ('Target', 'Prediction'), ('-', '--')):
        lgrads = dataset.longitudinal_gradients(data)
        std = lgrads.std().to('cpu').item()
        yp, xp = torch.histogram(lgrads, bins=args.n, density=True)
        xp = xp.to('cpu').numpy()
        xp = 0.5 *(xp[1:] + xp[:-1])
        yp = yp.to('cpu').numpy()
        axs[1].plot(xp / std, yp * std , style, color='black', label=label)
        axs[1].set_title('Longitudinal velocity gradient PDFs')
        axs[1].set_ylabel('$\sigma P(\partial u / \partial x)$')
        axs[1].set_xlabel('$(\partial u / \partial x)/\sigma$')
        axs[1].legend(loc='best')
    plt.savefig(os.path.join(args.out, 'hist.png'))
    
    return
