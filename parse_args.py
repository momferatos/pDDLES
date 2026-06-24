import torch
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='Template')

    # === PATHS === #

    parser.add_argument('-prefix',
                        type=str,
                        default='',
                        help='Output goes to $pDDLES/prefix')

    parser.add_argument('-predict',
                        action='store_true',
                        help='Prediction mode')

    parser.add_argument('-copy',
                        action='store_true',
                        help='Copy files to $LOCALSCRATCH')

    parser.add_argument('-drop_last',
                        action='store_false',
                        help='Drop remainder files in dataloaders')

    parser.add_argument('-noload',
                        action='store_true',
                        help='Don\'t load data to memory')

    parser.add_argument('-noskip',
                        action='store_true',
                        help='No skip connection')


    parser.add_argument('-scalar',
                        action='store_true',
                        help='Work with scalar data')

    parser.add_argument('-dev',
                        type=str,
                        default="gpu",
                        help='Device to use: gpu (default) or cpu')

    parser.add_argument('-device',
                        type=str,
                        default="cuda:0",
                        help='PyTorch device name')

    parser.add_argument('-data',
                        type=str,
                        default="data",
                        help='path to dataset directory')

    parser.add_argument('-out',
                        type=str,
                        default="out",
                        help='path to out directory')

    # === GENERAL === #
    parser.add_argument('-model',
                        type=str,
                        default="TEST",
                        help='Model name')

    parser.add_argument('-reset',
                        action='store_true',
                        help='Reset saved model logs and weights')

    parser.add_argument('-tb',
                        action='store_true',
                        help='Start TensorBoard')

    parser.add_argument('-gpus',
                        type=str,
                        default="0",
                        help='GPUs list, only works if not on slurm')

    parser.add_argument('-cfg',
                        type =str,
                        help='Configuration file')

    # === Dataset === #
    parser.add_argument('-dataset',
                        type=str,
                        default='TurbDataset',
                        help='Dataset to choose')

    parser.add_argument('-batch_per_task',
                        type=int,
                        default=32,
                        help='batch size per gpu')

    parser.add_argument('-shuffle',
                        type=bool,
                        default=True,
                        help='Shuffle dataset?')

    parser.add_argument('-workers',
                        type=int,
                        default=0,
                        help='number of workers')

    # === Architecture === #
    parser.add_argument('-arch',
                        type=str,
                        default='FNet',
                        help='Architecture')

    # === Trainer === #
    parser.add_argument('-trainer',
                        type=str,
                        default = 'trainer',
                        help='Trainer')

    parser.add_argument('-epochs',
                        type=int,
                        default=10,
                        help='Number of epochs')

    parser.add_argument('-save_every',
                        type=int,
                        default = 1,
                        help='Save frequency')

    parser.add_argument('-fp16',
                        action='store_true',
                        help='Use mixed precision.')


    # === Optimization === #
    parser.add_argument('-optimizer',
                        type=str,
                        default = 'adam',
                        help='Optimizer function to choose')

    parser.add_argument('-lr_start',
                        type=float,
                        default = 1e-2,
                        help='Initial Learning Rate')

    parser.add_argument('-lr_end',
                        type=float,
                        default = 1e-3,
                        help='Final Learning Rate')

    parser.add_argument('-lr_warmup',
                        type=int,
                        default=10,
                        help='Warmup epochs for learning rate')

    # === SLURM === #
    parser.add_argument('-slurm',
                        action='store_true',
                        help='Submit with slurm')

    parser.add_argument('-tasks_per_node',
                        type=int,
                        default=None,
                        help='Number of SLURM tasks per node')


    parser.add_argument('-nnodes',
                        type=int,
                        default=1,
                        help='Number of SLURM nodes')

    parser.add_argument('-nodelist',
                        default = None,
                        help='SLURM nodeslist. i.e. "GPU17,GPU18"')

    parser.add_argument('-partition',
                        type=str,
                        default="gpu",
                        help='SLURM partition')

    parser.add_argument('-account',
                        type=str,
                        default="p200140",
                        help='SLURM account')

    parser.add_argument('-timeout',
                        type=str,
                        default = '2-0:0:0',
                        help='SLURM timeout')

    parser.add_argument('-qos',
                        type=str,
                        default='default',
                        help='SLURM QoS')

    parser.add_argument('-mem_per_node',
                        type=int,
                        default=0,
                        help='SLURM memory per node')

    parser.add_argument('-mem',
                        type=str,
                        default='0',
                        help='SLURM memory')

    parser.add_argument('-num_blocks',
                        help='Number of blocks in the ResNet or FNet',
                        default=8,
                        type=int)

    parser.add_argument('-num_featmaps',
                        help='Number of CNN feature maps'
                        'in the ResNet or SuperNet',
                        default=8,
                        type=int)

    parser.add_argument('-kernel_size',
                        help='Size of the convolution kernel'
                        'in the ResNet or SuperNet',
                        default=3,
                        type=int)

    parser.add_argument('-dropout',
                        help='Dropout factor'
                        'in the ResNet or SuperNet',
                        default=0.2,
                        type=float)

    parser.add_argument('-num_coeffs',
                        help='Number of trainable spectral coefficients in' 
                        'each FourierBlock',
                        default=0,
                        type=int)

    parser.add_argument('-num_levels',
                        help='Number of Wavelet transform levels ',
                        default=None,
                        type=int)

    parser.add_argument('-dimensions',
                        help='Feature dimensions: 2 for image, 3 for'
                        'volumetric data.',
                        default=3,
                        type=int)

    parser.add_argument('-scaler',
                        help='Data normalization: \'norm\' or \'minmax\'',
                        default='norm',
                        type=str)

    parser.add_argument('-alpha',
                        help='Fraction of the wavenumber range to'
                        'be retained by the low-pass LES filter',
                        default=0.1,
                        type=float)

    parser.add_argument('-datafile',
                        help='Training/testing HDF5 filenames\' list file',
                        default='32.dat',
                        type=str)

    parser.add_argument('-hdf5_key',
                        help='HDF5 dataset key',
                        default='u',
                        type=str)


    parser.add_argument('-actfun',
                        help='PyTorch Activation function name',
                        default='ReLU',
                        type=str)


    parser.add_argument('-wavelet',
                        help='PyWavelet wavelet name',
                        default='db4',
                        type=str)

    parser.add_argument('-wavelet_mode',
                        help='Wavelet coefficient multiplication mode',
                        default='outer',
                        type=str)

    args = parser.parse_args()

    args.actfun = eval(f'torch.nn.{args.actfun}()')

    # === Read CFG File === #
    if args.cfg:
        with open(args.cfg, 'r') as f:
            import ruamel.yaml as yaml
            yml = yaml.safe_load(f)

        # update values from cfg file only if not passed in cmdline
        cmd = [c[1:] for c in sys.argv if c[0]=='-']
        for k,v in yml.items():
            if k not in cmd:
                args.__dict__[k] = v

    return args, parser
