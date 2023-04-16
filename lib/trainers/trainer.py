# Copyright (c) Ramy Mounir.
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

import torch
from lib.utils.file import checkdir
from lib.utils.tensorboard import get_writer, TBWriter
from lib.core.scheduler import cosine_scheduler, constant_scheduler
from lib.utils.distributed import MetricLogger
from glob import glob
import math

class dummy:
    def init(self):
        return
    
class Trainer:

    def __init__(self, args, train_loader, test_loader, model, loss, optimizer, dataset, scaler):

        self.args = args
        self.train_gen = train_loader
        self.test_gen = test_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.dataset = dataset
        self.fp16_scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
        self.test_losses = []
        self.train_losses = []
        self.device = args.device
                
        self.scaler = scaler
        
        # === TB writers === #
        if self.args.main:	

            self.writer = get_writer(args)
            self.lr_sched_writer = TBWriter(self.writer, 'scalar', 'Schedules/Learning Rate')			
            self.loss_writer = TBWriter(self.writer, 'scalar', 'Loss/total')

            checkdir("{}/weights/{}/".format(args.out, self.args.model), args.reset)


    def train_one_epoch(self, epoch, lr_schedule, args):

        metric_logger = MetricLogger(args, delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, self.args.epochs)
        
        test_div = 0.0
        train_div = 0.0
        train_loss = 0.0
        test_loss = 0.0
        for it, input_data in enumerate(metric_logger.log_every(self.train_gen, 10, args, header)):
            # === Global Iteration === #
            it = len(self.train_gen) * epoch + it

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]

            
            # === Inputs === #
            if args.device == 'gpu':
                input_data = input_data.cuda(non_blocking=True)
                autocast = torch.cuda.amp.autocast(self.args.fp16)
            else:
                autocast = torch.autocast(device_type='cpu')
                
            # === Forward pass === #
            with autocast:
                y = input_data.to(self.device)
                
                X = self.dataset.LES_filter(y)
                                
                # normalize
                X, y = self.scaler.transform(X, y, direction='forward')

                if not args.scalar:
                    Xh = self.dataset.to_helical(X)
                    preds = self.dataset.from_helical(self.model(Xh))
                else:
                    preds = self.model(X)
                
                labels = y
                loss = self.loss(preds, labels)
                train_loss += loss
                div = self.dataset.divergence(preds)
                train_div = max(div, train_div)

            # Sanity Check
            if not math.isfinite(loss.item()):
                print("Training loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)
            
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
            if args.device == 'gpu':
                torch.cuda.synchronize()
                
            metric_logger.update(train_loss=loss.item())
            metric_logger.update(train_div=train_div.item())

            if self.args.main:
                self.loss_writer(metric_logger.meters['train_loss'].value, it)
                self.loss_writer(metric_logger.meters['train_div'].value, it)
                self.lr_sched_writer(self.optimizer.param_groups[0]["lr"], it)


        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

        with torch.no_grad():
            test_metric_logger = MetricLogger(args, delimiter="  ")
            for it, input_data in enumerate(test_metric_logger.log_every(self.test_gen, 10, args, header)):

                # === Global Iteration === #
                it = len(self.train_gen) * epoch + it

                # === Inputs === #
                
                if args.device == 'gpu':
                    input_data = input_data.cuda(non_blocking=True)
                    autocast = torch.cuda.amp.autocast(self.args.fp16)
                else:
                    autocast = torch.autocast(device_type='cpu')
                    
                # === Forward pass === #
                with autocast:
                    y = input_data.to(self.device)
                    X = self.dataset.LES_filter(y)

                    # normalize
                    X, y = self.scaler.transform(X, y, direction='forward')

                    if not args.scalar:
                        Xh = self.dataset.to_helical(X)
                        preds = self.dataset.from_helical(self.model(Xh))
                    else:
                        preds = self.model(X)
                        
                    div = self.dataset.divergence(preds)
                    
                    labels = y
                    loss = self.loss(preds, labels)
                    test_loss += loss
                    test_div = max(div, test_div)

                # Sanity Check
                if not math.isfinite(loss.item()):
                    print("Test loss is {}, stopping training".format(loss.item()), force=True)
                    sys.exit(1)

                # === Logging === #
                if args.device == 'gpu':
                    torch.cuda.synchronize()
                test_metric_logger.update(test_loss=loss.item())
                test_metric_logger.update(test_div=div.item())

                if self.args.main:
                    self.loss_writer(test_metric_logger.meters['test_loss'].value, it)
                    self.loss_writer(test_metric_logger.meters['test_div'].value, it)

            test_metric_logger.synchronize_between_processes()
            print("Averaged stats:", test_metric_logger)

            self.test_losses.append(test_loss.item() / len(self.test_gen))
            self.train_losses.append(train_loss.item() / len(self.train_gen))

    def fit(self):

        # === Resume === #
        self.load_if_available()

        # === Schedules === #
        lr_schedule = constant_scheduler(
                        base_value = self.args.lr_start, # * (self.args.batch_per_task * self.args.world_size) / 256.,
                        final_value = self.args.lr_end,
                        epochs = self.args.epochs,
                        niter_per_ep = len(self.train_gen),
                        warmup_epochs= self.args.lr_warmup,
        )

        # === training loop === #
        for epoch in range(self.start_epoch, self.args.epochs):

            self.train_gen.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch, lr_schedule, self.args)

            # === save model === #
            if self.args.main and epoch%self.args.save_every == 0:
                self.save(epoch)

        
        print('Done.')
        
        return self.train_losses, self.test_losses
    
    def load_if_available(self):

        ckpts = sorted(glob(f'{self.args.out}/weights/{self.args.model}/Epoch_*.pth'))

        if len(ckpts) >0:
            ckpt = torch.load(ckpts[-1], map_location='cpu')
            self.start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if self.args.fp16: self.fp16_scaler.load_state_dict(ckpt['fp16_scaler'])
            print("Loaded ckpt: ", ckpts[-1])

        else:
            self.start_epoch = 0
            print("Starting from scratch")


    def save(self, epoch):

        if self.args.fp16:
            state = dict(epoch=epoch+1, 
                            model=self.model.module.state_dict(), 
                            optimizer=self.optimizer.state_dict(), 
                            fp16_scaler = self.fp16_scaler.state_dict(),
                            args = self.args,
                            means = (self.scaler.X_mean, self.scaler.y_mean),
                            stds = (self.scaler.X_std, self.scaler.y_std)
                        )
        else:
            state = dict(epoch=epoch+1, 
                            model=self.model.module.state_dict(), 
                            optimizer=self.optimizer.state_dict(),
                            args = self.args,
                            means = (self.scaler.X_mean, self.scaler.y_mean),
                            stds = (self.scaler.X_std, self.scaler.y_std)
                        )

            torch.save(state, "{}/weights/{}/Epoch_{}.pth".format(self.args.out, self.args.model, str(epoch).zfill(3) ))
