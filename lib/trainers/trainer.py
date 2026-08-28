# Copyright (c) Ramy Mounir.
# Copyright (c) 2022-2026 Georgios Momferatos.
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
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from lib.datasets.TurbDataset import TurbDataset
from lib.utils.file import checkdir
from lib.utils.tensorboard import get_writer, TBWriter
from lib.core.scheduler import get_scheduler
from lib.utils.distributed import MetricLogger, \
    is_dist_avail_and_initialized, cleanup_distributed
from glob import glob
import math
import sys


class Trainer:

    def __init__(self, args: argparse.Namespace, train_loader: DataLoader,
                 test_loader: DataLoader, valid_loader: DataLoader,
                 model: nn.Module, loss: nn.Module, optimizer: Optimizer,
                 dataset: TurbDataset, scaler: Any) -> None:

        self.args = args
        self.train_gen = train_loader
        self.test_gen = test_loader
        self.val_gen = valid_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.dataset = dataset
        self.scaler = scaler
        self.fp16_scaler = torch.amp.GradScaler('cuda') if args.fp16 else None
        self.test_losses = []
        self.train_losses = []
        self.val_loss = 0.0
        self.device = args.device
#        self.scaler.fit()
        
        # === TB writers === #
        if self.args.main:

            self.writer = get_writer(args)
            self.lr_sched_writer = TBWriter(
                self.writer, 'scalar', 'Schedules/Learning Rate')
            # one writer per series: funnelling every metric through a
            # single tag interleaves six series into one unreadable chart
            self.train_loss_writer = TBWriter(
                self.writer, 'scalar', 'Loss/train')
            self.test_loss_writer = TBWriter(
                self.writer, 'scalar', 'Loss/test')
            self.val_loss_writer = TBWriter(
                self.writer, 'scalar', 'Loss/val')
            self.train_div_writer = TBWriter(
                self.writer, 'scalar', 'Divergence/train')
            self.test_div_writer = TBWriter(
                self.writer, 'scalar', 'Divergence/test')
            self.val_div_writer = TBWriter(
                self.writer, 'scalar', 'Divergence/val')

            checkdir("{}/weights/{}/".format(args.out, self.args.model), args.reset)


    def _global_max(self, value: float) -> float:
        """Reduce a scalar to its global maximum across all ranks.

        Divergence is tracked as a per-rank running max; this collapses it to
        the true max over the whole distributed batch. No-op without dist.
        """
        if not is_dist_avail_and_initialized():
            return float(value)
        t = torch.as_tensor(float(value), dtype=torch.float64,
                             device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return t.item()

    def _stop_if_nonfinite(self, value: float, what: str) -> None:
        """Collectively stop every rank if any rank's loss is non-finite.

        A bare sys.exit on the offending rank alone leaves the other ranks
        blocked in their next collective until the backend times out; the
        MAX all-reduce makes every rank take the same branch together, so
        they can tear down the process group and exit in unison.
        """
        flag = 0.0 if math.isfinite(value) else 1.0
        if is_dist_avail_and_initialized():
            t = torch.tensor(flag, dtype=torch.float64, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            flag = t.item()
        if flag > 0:
            print(f'{what} loss is non-finite on some rank '
                  f'(this rank: {value}), stopping training', force=True)
            cleanup_distributed()
            sys.exit(1)

    def train_one_epoch(self, epoch: int, lr_schedule: np.ndarray,
                        args: argparse.Namespace) -> None:

        metric_logger = MetricLogger(args, delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, self.args.epochs)
        
        test_div = 0.0
        train_div = 0.0
        for it, input_data in enumerate( \
                metric_logger.log_every(self.train_gen, 10, args, header)):
            # === Global Iteration === #
            it = len(self.train_gen) * epoch + it

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]

            
            # === Inputs === #
            if args.dev == 'gpu':
                input_data = input_data.cuda(non_blocking=True)
                autocast = torch.amp.autocast('cuda', enabled=self.args.fp16)
            else:
                autocast = torch.autocast(device_type='cpu')
                
            # === Forward pass === #
            with autocast:
                y = input_data.to(self.device)
                                
                X = self.dataset.LES_filter(y)
                                
                # normalize
                X, y = self.scaler.transform(X, y, action='scale')

                preds = self.model(X)

                labels = y
                
                loss = self.loss(preds, labels)

                # .item(): div is logged only, never backpropped; a plain
                # float cannot pin preds' autograd graph across iterations
                # and keeps the running max type-stable.
                div = self.dataset.divergence(preds).item()
                train_div = max(div, train_div)

            # Sanity Check
            self._stop_if_nonfinite(loss.item(), 'Training')
            
            # === Backward pass === #
            self.model.zero_grad()

            if self.args.fp16:
                self.fp16_scaler.scale(loss).backward()
                self.fp16_scaler.step(self.optimizer)
                self.fp16_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()


            # === Logging === #
            if args.dev == 'gpu':
                torch.cuda.synchronize()
                
            metric_logger.update(train_loss=loss.item())
            metric_logger.update(train_div=train_div)

            if self.args.main:
                self.train_loss_writer(
                    metric_logger.meters['train_loss'].value, it)
                self.train_div_writer(
                    metric_logger.meters['train_div'].value, it)
                self.lr_sched_writer(self.optimizer.param_groups[0]["lr"], it)


        # reduce the per-rank running max to the global max across ranks
        metric_logger.update(train_div=self._global_max(train_div))
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

        with torch.no_grad():
            test_metric_logger = MetricLogger(args, delimiter="  ")
            for it, input_data in enumerate( \
                    test_metric_logger.log_every(self.test_gen, 10, args,
                                                 header)):

                # === Global Iteration === #
                # scale by this loader's own length: train_gen's length put
                # successive epochs' test points on overlapping steps
                it = len(self.test_gen) * epoch + it

                # === Inputs === #

                if args.dev == 'gpu':
                    input_data = input_data.cuda(non_blocking=True)
                    autocast = torch.amp.autocast('cuda', enabled=self.args.fp16)
                else:
                    autocast = torch.autocast(device_type='cpu')
                    
                # === Forward pass === #
                with autocast:
                    y = input_data.to(self.device)
                    
                    X = self.dataset.LES_filter(y)
                    
                    # normalize
                    X, y = self.scaler.transform(X, y, action='scale')

                
                    preds = self.model(X)

                    
                    div = self.dataset.divergence(preds)
                    
                    labels = y
                    
                    loss = self.loss(preds, labels)
                    test_div = max(div.item(), test_div)
                # Sanity Check
                self._stop_if_nonfinite(loss.item(), 'Test')

                # === Logging === #
                if args.dev == 'gpu':
                    torch.cuda.synchronize()
                test_metric_logger.update(test_loss=loss.item())
                test_metric_logger.update(test_div=div.item())

                if self.args.main:
                    self.test_loss_writer(
                        test_metric_logger.meters['test_loss'].value, it)
                    self.test_div_writer(
                        test_metric_logger.meters['test_div'].value, it)

            # reduce the per-rank running max to the global max across ranks
            test_metric_logger.update(test_div=self._global_max(test_div))
            test_metric_logger.synchronize_between_processes()
            print("Averaged stats:", test_metric_logger)

            # record the epoch means: after synchronize_between_processes
            # global_avg is the average over every batch on every rank,
            # whereas .value is just this rank's last batch - far too noisy
            # for the loss curves and the best-epoch selection built on them
            self.test_losses.append(
                test_metric_logger.meters['test_loss'].global_avg)
            self.train_losses.append(
                metric_logger.meters['train_loss'].global_avg)

    def fit(self) -> tuple[list[float], list[float], float]:

        # === Resume === #
        self.load_if_available()

        # === Schedules === #
        lr_schedule = get_scheduler(self.args, len(self.train_gen))

        # === training loop === #
        # When resuming a run that already reached args.epochs the loop body
        # never executes, but the validation pass below still needs `epoch`
        # for its header and global step; give it the last-epoch value.
        epoch = max(self.args.epochs - 1, 0)
        for epoch in range(self.start_epoch, self.args.epochs):

            self.train_gen.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch, lr_schedule, self.args)

            # === save model === #
            if self.args.main and epoch%self.args.save_every == 0:
                self.save(epoch)

        print('Calculating validation loss...')
        with torch.no_grad():
            val_div = 0.0
            header = 'Epoch: [{}/{}]'.format(epoch, self.args.epochs)
            val_metric_logger = MetricLogger(self.args, delimiter="  ")
            for it, input_data in enumerate( \
                    val_metric_logger.log_every( \
                        self.val_gen, 10, self.args, header)):

                # === Global Iteration === #
                it = len(self.val_gen) * epoch + it

                # === Inputs === #

                if self.args.dev == 'gpu':
                    input_data = input_data.cuda(non_blocking=True)
                    autocast = torch.amp.autocast('cuda', enabled=self.args.fp16)
                else:
                    autocast = torch.autocast(device_type='cpu')
                    
                # === Forward pass === #
                with autocast:
                    y = input_data.to(self.device)
                    
                    X = self.dataset.LES_filter(y)
                    
                    # normalize
                    X, y = self.scaler.transform(X, y, action='scale')

                
                    preds = self.model(X)
                        
                    div = self.dataset.divergence(preds)
                    
                    labels = y
                    
                    loss = self.loss(preds, labels)
                    val_div = max(div.item(), val_div)
                # Sanity Check
                self._stop_if_nonfinite(loss.item(), 'Val')

                # === Logging === #
                if self.args.dev == 'gpu':
                    torch.cuda.synchronize()
                val_metric_logger.update(val_loss=loss.item())
                val_metric_logger.update(val_div=div.item())

                if self.args.main:
                    self.val_loss_writer(
                        val_metric_logger.meters['val_loss'].value, it)
                    self.val_div_writer(
                        val_metric_logger.meters['val_div'].value, it)

            # reduce the per-rank running max to the global max across ranks
            val_metric_logger.update(val_div=self._global_max(val_div))
            val_metric_logger.synchronize_between_processes()
            print("Averaged stats:", val_metric_logger)
            # epoch mean across all ranks and batches, not the last batch
            self.val_loss = val_metric_logger.meters['val_loss'].global_avg
            
        print('Done.')
        
        return self.train_losses, self.test_losses, self.val_loss

    
    def load_if_available(self) -> None:

        ckpts = sorted(
            glob(f'{self.args.out}/weights/{self.args.model}/Epoch_*.pth'))

        if len(ckpts) >0:
            # weights_only=False: our checkpoints embed `args` (numpy scalars,
            # the activation nn.Module), which the PyTorch>=2.6 safe unpickler
            # rejects. Safe here since we write these checkpoints ourselves.
            ckpt = torch.load(ckpts[-1], map_location='cpu', weights_only=False)
            self.start_epoch = ckpt['epoch']
            self.model.module.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.args.fp16: self.fp16_scaler.load_state_dict(
                    ckpt['fp16_scaler'])
            if 'scaler' in ckpt:
                # normalization constants travel inside the checkpoint
                self.scaler.load_state_dict(ckpt['scaler'])
                print("Loaded ckpt: ", ckpts[-1])
            else:
                # checkpoint predates embedded scaler constants: refit them
                print("Loaded ckpt (no scaler state), refitting scaler:",
                      ckpts[-1])
                self.scaler.fit()

        else:
            self.start_epoch = 0
            print("Starting from scratch; fitting scaler")
            self.scaler.fit()


    def save(self, epoch: int) -> None:

        # normalization constants ride in the checkpoint (self.scaler.state_dict)
        # instead of a separate norm.pt file
        state = dict(epoch=epoch+1,
                        model=self.model.module.state_dict(),
                        optimizer=self.optimizer.state_dict(),
                        scaler=self.scaler.state_dict(),
                        args = self.args
                    )
        if self.args.fp16:
            state['fp16_scaler'] = self.fp16_scaler.state_dict()

        torch.save(state,
                   "{}/weights/{}/Epoch_{}.pth".format(self.args.out,
                                                       self.args.model,
                                                       str(epoch).zfill(3)
                   ))
