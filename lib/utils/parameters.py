######################################################
# DDLES: Data-driven model for Large Eddy Simulation #
# Georgios Momferatos, 2022-2023                     #
# g.momferatos@ipta.demokritos.gr                    #
######################################################

import argparse
import importlib
import os
import torch
import torch.nn as nn
from lib.utils.global_variables import device

from lib.LossFunctions import SubgridLoss

class parameters:
    """Global parameters object
    
    Parameters
    ----------
    None

    Attributes
    ----------
    
    params:
       Dictionary holding global parameters

    """
    
    def __init__(self):
        self.params = dict()
        return

    def set_defaults(self):
        """Set defalut parameters

            Parameters
            ----------
            None


            Returns
            ----------
            None

        """

        # activate CUDA GPU acceleration
        self.params["device"] = 'cuda:0'
        
        self.params['n'] = None # Global variable for the size of
                           # the DNS data square/cube
        # Feature dimensions (2 for image, 3 for volumetric data)
        self.params['dimensions'] = 2
        self.params['norm_mode'] = 'mean_std'
        # fraction of the wavenumber range to be retained by the low-pass filter
        self.params['alpha'] = 0.1 
        self.params['epochs'] = 10 # number of epochs
        self.params['batch_size'] = 512 # size of the minbatch
        self.params['num_workers'] = 1 # size of the minbatch
        self.params['num_blocks'] = 8 # Number of blocks in the model  
        self.params['num_channels'] = 8 # Number of channels in the ResNet block
        self.params["kernel_size"] = 3 # Size of the convolution kernel
        self.params["dropout"] = None # Dropout factor for ResNet & Supernet
        # Number of coefficients in each FourierBlock
        self.params['num_coeffs'] = 128
        # Number of Discrete Wavelet Transform Levels
        self.params['num_levels'] = 3
        # Type of model: ('ResNet', 'FourierNet' 'WaveletNet' or 'SuperNet')
        self.params['model_type'] = 'ResNet' 
        # Prediction mode: 'large_to_all' learns the map X -> y with
        # X = LES_filter(y) i.e. from large scales to all scales
        # 'large_to_small' the map X -> y' with X = LES_filter(y) and
        # y' = y - LES_filter(y) i.e. directly from
        # large scales to the small scales
        self.params['prediction_mode'] = 'large_to_all'
        # Nonlinear activation function to be used
        self.params['actfun'] = nn.ReLU() 
        self.params['nn_conv'] = None # Convolution component
        self.params['nn_batchnorm'] = None # Batch normalization component
        # torch.optim optimizer to be used for training
        self.params['optimizer'] = torch.optim.Adam
        # Dictionary containing torch.optim optimizer options
        self.params['optimizer_options'] = dict()
        # torch.optim.lr_scheduler learning rate scheduler to be used for
        # training
        self.params['scheduler'] = None
        # Dictionary containing torch.optim.ler_scheduler learning rate
        # scheduler options
        self.params['scheduler_options'] = dict() 
        self.params['learning_rate'] = 1.0e-3 # learning rate
        self.params['hdf5_key'] = 'scl' # HDF5 input key
        # default training/testing HDF5 filenames' list file
        self.params['train_test_file'] = 'train_test.dat'
        # default prediction HDF5 filenames' list file
        self.params['prediction_file'] = 'predict.dat'
        # Operation mode: 'train', 'test' or 'both'
        self.params['operation_mode'] = 'train'
        # PyTorch model filename to save/load' 
        self.params['model_filename'] = 'foo'
        # name of the training session
        self.params["session_name"] = 'foo' 
        self.params['wavelet_type'] = 'db4' # Wavelet type
        # Wavelet coefficient multiplication mode: ('one', 'outer' or 'all')
        self.params['wavelet_mode']='outer'
        self.params['loss_function'] = torch.nn.MSELoss # Loss function

        return 

    def class_for_name(self, module_name, class_name):
        """Get the class of name "class_name" contained in a module 
        "module_name"
    
        Parameters
        ----------
        module_name: str
           Name of the module containing the clas

        class_name: str
           Name of the class

        Returns
        ----------
        class: Python object
            The class object of name "class_name" contained in a module 
        "module_name"

        """
        # load the module, will raise ImportError if module cannot be loaded
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        c = getattr(m, class_name)

        return c

    def apply_cmd_args(self, parser):
        """Apply command line arguments

            Parameters
            ----------
            None

            Returns
            ----------
            None

        """

        # parse the command line
        args = self.cmd_args(parser)

        # apply command line argumemts
        for key,value in args.__dict__.items():
            if value:
                if key == 'optim':
                    self.params[key] = self.class_for_name('torch.optim',
                                                              value)
                elif key == 'optimizer_options':
                    self.params[key] = eval(value)
                elif key == 'scheduler':
                    self.params[key] = self.class_for_name(
                        'torch.optim.lr_scheduler', value)
                elif key == 'scheduler_options':
                    self.params[key] = eval(value)
                elif key == 'loss_function':
                    if value == 'SubgridLoss':
                        continue
                    else:
                        self.params[key] = self.class_for_name('torch.nn',
                                                                   value)
                elif key == 'actfun':
                    self.params[key] = eval(f'torch.nn.{value}()')
                else:
                    self.params[key] = value

        device = self.params["device"]
        
        return




    def write_print(self):
        """Write global parameters

            Parameters
            ----------
            None

            Returns
            ----------
            None

        """

        # write

        with open(os.path.join(f'{self.params["session_name"]}',
                               'params.dat'), 'w') as f:
            f.write(f'Model type: {self.params["model_type"]}.\n')
            if (self.params["model_type"] == 'ResNet'
                or self.params["model_type"] == 'FourierNet'
                or self.params["model_type"] == 'WaveletNet'):
                  f.write(f'Number of blocks: {self.params["num_blocks"]}.\n')
            if (self.params["model_type"] == 'ResNet'
                or self.params["model_type"]== 'SuperNet'):
                  f.write('Number of convolutional channels: '
                          f'{self.params["num_channels"]}.\n')
            if self.params["model_type"] == 'WaveleNet':
                  f.write('Number of Wavelet transform levels:'
                          f'{self.params["num_levels"]}.\n')
                  f.write('Wavelet parameter type: '
                          f'{self.params["wavelet_type"]}\n')
                  f.write('Wavelet parameter multiplication mode: '
                          f'{self.params["wavelet_mode"]}\n')
            f.write(f'Activation function: {self.params["actfun"]}.\n')
            f.write(f'Feature dataset dimensions: '
                    f'{self.params["dimensions"]}d.\n')
            f.write(f'Prediction mode: {self.params["prediction_mode"]}.\n')
            f.write(f'Normalization mode: {self.params["norm_mode"]}.\n')
            f.write(f'alpha: {self.params["alpha"]}.\n')
            f.write('\n')
            f.write(f'Learning rate: {self.params["learning_rate"]:.4e}.\n')
            optim_name = self.params["optimizer"].__name__
            f.write(f'Optimizer: {optim_name}.\n')
            if self.params["optimizer_options"]:
                f.write(f'Optimizer options: \n')
                for key,value in self.params["optimizer_options"].items():
                    f.write(f'   {key} = {value}\n')
            if self.params["scheduler"]:
                scheduler_name = self.params["scheduler"].__name__
                f.write(f'Learning rate scheduler: {scheduler_name}.\n')
            if self.params["scheduler_options"]:
                f.write(f'Learning rate scheduler options: \n')
                for key,value in self.params["scheduler_options"].items():
                    f.write(f'   {key} = {value}\n')
            f.write(f'Epochs: {self.params["epochs"]}.\n')
            f.write(f'Minibatch size: {self.params["batch_size"]}.\n')
            f.write(f'Loss function: {self.params["loss_function"].__name__}\n')
            f.write(f'HDF5 key: {self.params["hdf5_key"]}.\n')
            if self.params["operation_mode"] != 'predict':
                f.write('Train/test HDF5 filenames\' list file: '
                    f'{self.params["train_test_file"]}.\n')
            if self.params["operation_mode"] != 'train':
                f.write(
                'Prediction HDF5 filenames\' list file: '
                    f'{self.params["prediction_file"]}.\n')
            f.write(f'Operation mode: {self.params["operation_mode"]}\n')
            f.write(f'Session name: {self.params["session_name"]}\n')
            tmp = os.path.join(self.params["session_name"],
                               self.params["model_filename"]) + '.pt'
            f.write(f'Model filename: {tmp}\n')


        # print
        with open(os.path.join(f'{self.params["session_name"]}',
                               'params.dat'), 'r') as f:
            print(f.read())

        return

    def cmd_args(self, parser):
        """Get command line arguments

            Parameters
            ----------
            None

            Returns
            ----------
            None

        """

        # parser.add_argument('--epochs',
        #                     help='Number of training epochs (default: 10)',
        #                     type=int)

        # parser.add_argument('--batch_size',
        #                     help='Size of the minibatch (default: 512)',
        #                     type=int)

        # parser.add_argument('--num_workers',
        #                     help='Number of Dataloader workers (default: 1)',
        #                     type=int)

        
        # parser.add_argument('--num_blocks',
        #                     help='Number of blocks in the ResNet or FourierNet'
        #                     '(default: 8)',
        #                     type=int)

        # parser.add_argument('--num_channels',
        #                     help='Number of convolutional channels'
        #                     'in the ResNet or SuperNet'
        #                     '(default: 8)',
        #                     type=int)

        # parser.add_argument('--kernel_size',
        #                     help='Size of the convolution kernel'
        #                     'in the ResNet or SuperNet'
        #                     '(default: 3)',
        #                     type=int)

        # parser.add_argument('--dropout',
        #                     help='Dropout factor'
        #                     'in the ResNet or SuperNet'
        #                     '(default: 0.2)',
        #                     type=float)

        # parser.add_argument('--num_coeffs',
        #                     help='Number of trainable spectral coefficients in' 
        #                     'each FourierBlock'
        #                     '(default: 128)',
        #                     type=int)

        # parser.add_argument('--num_levels',
        #                     help='Number of Wavelet transform levels '
        #                     '(default: 3)',
        #                     type=int)


        # parser.add_argument('--loss_function',
        #                     help='Loss function (\'MSE\', \'L1\' or'
        #                     '\'Spectral\', (default: \'MSE\')'
        #                     '(default: \'MSE\')',
        #                     type=str)

        # parser.add_argument('--dimensions',
        #                     help='Feature dimensions: 2 for image, 3 for'
        #                     'volumetric data. (default: 2)',
        #                     type=int)
        # parser.add_argument('--norm_mode',
        #                     help='Data normalization: \'mean_std\' or \'min_max\''
        #                     '(default: \'mean_std\')',
        #                     type=str)
        # parser.add_argument('--alpha',
        #                     help='Fraction of the wavenumber range to'
        #                     'be retained by the low-pass LES filter'
        #                     '(default: 0.1)',
        #                     type=float)
        # parser.add_argument('--train_test_file',
        #                     help='Training/testing HDF5 filenames\' list file'
        #                     '(default: \'train_test_filenames2d.dat\')',
        #                     type=str)
        # parser.add_argument('--prediction_file',
        #                     help='Prediction HDF5 filenames\' list file'
        #                     '(default: \'prediction_filenames2d.dat\')',
        #                     type=str)
        # parser.add_argument('--hdf5_key',
        #                     help='HDF5 dataset key (default: \'ww\')',
        #                     type=str)
        # parser.add_argument('--learning_rate',
        #                     help='Learning rate (default: 1.0e-3)',
        #                     type=float)

        # parser.add_argument('--optimizer',
        #                     help='torch.optim optimizer to use. '
        #                     '(default: Adam)',
        #                     type=str)

        # parser.add_argument('--optimizer_options',
        #                     help='Dictionary of torch.optim optimizer '
        #                     'options'
        #                     '(default: None)',
        #                     type=str)

        # parser.add_argument('--scheduler',
        #                     help='torch.optim.lr_scheduler learning rate '
        #                     'scheduler '
        #                     'to use. '
        #                     '(default: None)',
        #                     type=str)

        # parser.add_argument('--scheduler_options',
        #                     help='Dictionary of torch.optim.lr_scheduler '
        #                     'learning rate scheduler options'
        #                     '(default: None)',
        #                     type=str)

        # parser.add_argument('--actfun',
        #                     help='Activation function to use:'
        #                     'Tanh, Sigmoid or ReLU'
        #                     '(default: Tanh)',
        #                     type=str)
        # parser.add_argument('--prediction_mode',
        #                     help='Prediction mode:'
        #                     '\'large to all\' or \'large_to_small\''
        #                     '(default: \'large to all\')\n'
        #                     '\'large_to_all\' learns the map X -> y with'
        #                     'X = LES_filter(y) i.e. from large scales to all'
        #                     'scales.\n'
        #                     '\'large_to_small\' learns the map'
        #                     'X -> y\' with X = LES_filter(y) and'
        #                     'y\' = y - LES_filter(y) i.e. directly from'
        #                     'the large scales to the small scales',
        #                     type=str)
        # parser.add_argument('--model_type',
        #                     help='Type of model (\'ResNet\', \'FourierNet\' or '
        #                     '\'SuperNet\')'
        #                     '(default: \'SuperNet\')',
        #                     type=str)
        # parser.add_argument('--operation_mode',
        #                     help='Operation mode: \'train\', \'predict\' '
        #                     'or \'both\''
        #                     '(default: \'both\')',
        #                     type=str),
        # parser.add_argument('--model_filename',
        #                     help='Filename in which to load/save the model'
        #                     '(default: \'session_name/foo.pt\')',
        #                     type=str)
        # parser.add_argument('--session_name',
        #                     help='Name of the training session '
        #                     '(default: \'foo\')',
        #                     type=str)
        # parser.add_argument('--wavelet_type',
        #                     help='Pywavelet wavelet type to use.'
        #                     '(default: \'db4\')',
        #                     type=str)
        # parser.add_argument('--wavelet_mode',
        #                     help='Wavelet coefficient multiplication mode'
        #                     '(default: \'outer\')',
        #                     type=str)

        args = parser.parse_args()

        return args
