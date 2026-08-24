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

import numpy as np

# copy-paste from https://github.com/facebookresearch/dino/blob/main/utils.py
def cosine_scheduler(base_value: float, final_value: float, epochs: int,
                     niter_per_ep: int, warmup_epochs: int = 0,
                     start_warmup_value: float = 0) -> np.ndarray:
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value,
                                          base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = (final_value + 0.5 * (base_value - final_value) *
                (1 + np.cos(np.pi * iters / len(iters))))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def warmup_scheduler(base_value: float, final_value: float, epochs: int,
                     niter_per_ep: int, warmup_epochs: int = 10) -> np.ndarray:

    warmup_schedule = np.linspace(base_value, final_value,
                                  warmup_epochs * niter_per_ep)
    schedule = np.ones((epochs - warmup_epochs) * niter_per_ep) * final_value

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def constant_scheduler(base_value: float, final_value: float, epochs: int,
                       niter_per_ep: int, warmup_epochs: int = 10) -> np.ndarray:

    schedule = np.linspace(base_value, base_value, epochs * niter_per_ep)
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def get_scheduler(args: argparse.Namespace, niter_per_ep: int) -> np.ndarray:
    """Build the learning-rate schedule selected by -scheduler.

    'constant' holds lr_start (the historical default; -lr_end/-lr_warmup
    have no effect). 'cosine' warms up over lr_warmup epochs then cosine-
    decays lr_start -> lr_end. 'warmup' ramps lr_start -> lr_end over
    lr_warmup epochs then holds lr_end.
    """
    schedulers = {
        'constant': constant_scheduler,
        'cosine': cosine_scheduler,
        'warmup': warmup_scheduler,
    }
    if args.scheduler not in schedulers:
        raise ValueError(
            "Unknown scheduler '{}'; choose from {}".format(
                args.scheduler, sorted(schedulers)))

    return schedulers[args.scheduler](
        base_value=args.lr_start,
        final_value=args.lr_end,
        epochs=args.epochs,
        niter_per_ep=niter_per_ep,
        warmup_epochs=args.lr_warmup,
    )
