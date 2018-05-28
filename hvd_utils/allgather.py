from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from horovod.common import init
from horovod.common import size
from horovod.common import local_size
from horovod.common import rank
from horovod.common import local_rank
from horovod.common import mpi_threads_supported
from horovod.common import check_extension

#check_extension('horovod.torch', 'HOROVOD_WITH_PYTORCH',
#                __file__, 'mpi_lib', '_mpi_lib')

from horovod.torch.mpi_ops import allreduce, allreduce_async, allreduce_, allreduce_async_
from horovod.torch.mpi_ops import allgather, allgather_async
from horovod.torch.mpi_ops import broadcast, broadcast_async, broadcast_, broadcast_async_
from horovod.torch.mpi_ops import poll, synchronize
import numpy as np
from ..prune_utils.pruning import select_top_k, select_top_k_appr, check_sparsity

import torch


class allreduce_DistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters=None):
        super(self.__class__, self).__init__(params)

        if named_parameters is not None:
            named_parameters = list(named_parameters)
        else:
            named_parameters = []

        # make sure that named_parameters are tuples
        if any([not isinstance(p, tuple) for p in named_parameters]):
            raise ValueError('named_parameters should be a sequence of '
                             'tuples (name, parameter), usually produced by '
                             'model.named_parameters().')

        self._parameter_names = {v: k for k, v
                                 in sorted(named_parameters)}
        self._handles = {}
        self._grad_accs = []

        if size() > 1:
            self._register_hooks()

    def _register_hooks(self):
        for param_group in self.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    p_tmp = p.expand_as(p)
                    grad_acc = p_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_hook(p))
                    self._grad_accs.append(grad_acc)

    def _make_hook(self, p):
        def hook(*ignore):
            assert p not in self._handles
            assert not p.grad.requires_grad
            name = self._parameter_names.get(p)
            p_size = np.prod(p.size())
            if p_size < 1024:
                handle = allreduce_async_(p.grad.data, average=True, name=name)
            else:
                masks = [torch.zeros(p.size()).cuda()]
                masks, compressed_val, compressed_idx = select_top_k_appr(p.grad, 0.001, masks)
                mgs = torch.cat([compressed_idx.type('torch.cuda.FloatTensor'), compressed_val])
                handle = hvd.allgather_async(msg)
                for node_idx in range(hvd.size()):
                    p.grad.data[msg[node_idx*msg_size*2 : node_idx*msg_size*2 + msg_size].type('torch.cuda.LongTensor')] += \
                    msg[node_idx*msg_size*2 + msg_size : node_idx*msg_size*2 + 2*msg_size]

            self._handles[p] = handle
        return hook

    def synchronize(self):
        for handle in self._handles.values():
            synchronize(handle)
        self._handles.clear()

    def step(self, closure=None):
        self.synchronize()
        return super(self.__class__, self).step(closure)


def myDistributedOptimizer(optimizer, named_parameters=None):
    """
    An optimizer that wraps another torch.optim.Optimizer, using an allreduce to
    average gradient values before applying gradients to model weights.
    Allreduce operations are executed after each gradient is computed by `loss.backward()`
    in parallel with each other. The `step()` method ensures that all allreduce operations are
    finished before applying gradients to the model.
    DistributedOptimizer exposes the `synchronize()` method, which forces allreduce operations
    to finish before continuing the execution. It's useful in conjunction with gradient
    clipping, or other operations that modify gradients in place before `step()` is executed.
    Example of gradient clipping:
    ```
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.synchronize()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()
    ```
    Arguments:
        optimizer: Optimizer to use for computing gradients and applying updates.
        named_parameters: A mapping between parameter names and values. Used for naming of
                          allreduce operations. Typically just `model.named_parameters()`.
    """
    # We dynamically create a new class that inherits from the optimizer that was passed in.
    # The goal is to override the `step()` method with an allreduce implementation.
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
               dict(allreduce_DistributedOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters)

