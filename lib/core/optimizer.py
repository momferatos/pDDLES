# Copyright (c) 2022-2026 Georgios Momferatos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse

import torch.nn as nn
import torch.optim as Opt

def get_optimizer(model: nn.Module, args: argparse.Namespace) -> Opt.Optimizer:

    # classes, not instances: the dict-of-instances form built all three
    # optimizers on every call and, on an unknown name, returned the string
    # "Invalid Optimizer" that then blew up at optimizer.param_groups.
    opt_fns = {
        'adam': Opt.Adam,
        'sgd': Opt.SGD,
        'adagrad': Opt.Adagrad,
    }
    if args.optimizer not in opt_fns:
        raise ValueError(
            "Unknown optimizer '{}'; choose from {}".format(
                args.optimizer, sorted(opt_fns)))

    optargs = {'lr': args.lr_start,
                'weight_decay': args.weight_decay}
    if args.optimizer == 'sgd' and args.momentum is not None:
        optargs['momentum'] = args.momentum
    if args.optim_options is not None:
        # Parse the additional optimizer options from the string
        additional_options = {}
        for option in args.optim_options.split(','):
            key, value = option.split('=')
            additional_options[key.strip()] = eval(value.strip())
        optargs.update(additional_options)
    return opt_fns[args.optimizer](model.parameters(),**optargs)