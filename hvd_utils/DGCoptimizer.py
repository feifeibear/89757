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
from horovod.torch.mpi_ops import allgather, allgather_async, _allgather_async
from horovod.torch.mpi_ops import broadcast, broadcast_async, broadcast_, broadcast_async_
from horovod.torch.mpi_ops import poll, synchronize
import numpy as np
from .pruning import select_top_k, select_top_k_appr, check_sparsity, prune_perc
import horovod.torch as hvd

import torch


class _DGCOptimizer(torch.optim.Optimizer):
    def __init__(self, params, named_parameters=None, use_gpu=True, momentum=0.9, weight_decay=1e-4):
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
        self._use_gpu = use_gpu
        self._use_nesterov = True
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._debug = False
        self._use_allgather= True

        # define U for residue, V for momentum
        if self._use_gpu:
            self._V = {k: torch.zeros(v.size()).cuda() for k, v
                                     in sorted(named_parameters)}
            self._U = {k: torch.zeros(v.size()).cuda() for k, v
                                     in sorted(named_parameters)}
            self._U = {k: torch.zeros(v.size()).cuda() for k, v
                                     in sorted(named_parameters)}
            self._masks = {k: torch.zeros(v.size()).cuda() for k, v
                                     in sorted(named_parameters)}
            self._compressed_msg = {k: torch.zeros(0).cuda() for k, v
                                 in sorted(named_parameters)}
        else:
            self._V = {k: torch.zeros(v.size()) for k, v
                                     in sorted(named_parameters)}
            self._U = {k: torch.zeros(v.size()) for k, v
                                     in sorted(named_parameters)}
            self._U = {k: torch.zeros(v.size()) for k, v
                                     in sorted(named_parameters)}
            self._masks = {k: torch.zeros(v.size()) for k, v
                                     in sorted(named_parameters)}
            self._compressed_msg = {k: torch.zeros(0) for k, v
                                 in sorted(named_parameters)}
        self._compressed_msg_size = {k: 0 for k, v
                                 in sorted(named_parameters)}
        self._v_ref = {k: [] for k, v
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
            if self._use_allgather and p_size > 1024:
                # fjr compress grad
                p.grad.data.add_(torch.mul(p.data, self._weight_decay))
                p.grad.data.div_(hvd.size())
                if self._use_nesterov:
                    self._U[name] = torch.mul(torch.add(self._U[name], p.grad.data), self._momentum)
                    self._V[name] = self._V[name] + self._U[name] + p.grad.data
                else:
                    self._U[name] = self._momentum * self._U[name] + p.grad.data
                    self._V[name] = self._V[name] + self._U[name]
                self._masks[name], compressed_val, compressed_idx = select_top_k_appr(self._V[name], 0.001, self._masks[name])
                if self._debug:
                    self._v_ref[name] = self._V[name] * self._masks[name]
                    allreduce_(self._v_ref[name], average = False)

                #self._V[name] = self._V[name] * (1 - self._masks[name])
                #self._U[name] = self._U[name] * (1 - self._masks[name])
                self._V[name].mul_(self._masks[name])
                self._U[name].mul_(self._masks[name])
                self._compressed_msg_size[name] = len(compressed_idx)
                if self._use_gpu:
                    compressed_msg = torch.cat([compressed_idx.type('torch.cuda.FloatTensor'), compressed_val])
                else:
                    compressed_msg = torch.cat([compressed_idx.type('torch.FloatTensor'), compressed_val])

                handle = _allgather_async(compressed_msg, self._compressed_msg[name], name=name)
                self._handles[p] = handle
            else:
                p.grad.data.add_(torch.mul(p.data, self._weight_decay))
                if self._use_nesterov:
                    self._U[name] = torch.mul(torch.add(self._U[name], p.grad.data), self._momentum)
                    self._V[name] = self._V[name] + self._U[name] + p.grad.data
                else:
                    self._U[name] = self._momentum * self._U[name] + p.grad.data
                    self._V[name] = self._V[name] + self._U[name]
                p.grad.data = self._V[name]
                handle = allreduce_async_(p.grad.data, average=True, name=name)
                self._handles[p] = handle

        return hook

    def synchronize(self):
        for p in self._handles:
            handle = self._handles[p]
            synchronize(handle)
            p_size = np.prod(p.size())
            if self._use_allgather and p_size > 1024:
                #fjr decompress
                name = self._parameter_names.get(p)
                msg_size = self._compressed_msg_size[name]
                #print("rank, msg_size is ", hvd.local_rank(), msg_size)
                g_size = p.grad.data.size()
                p_flatten = p.grad.data.view(-1)
                p_flatten.zero_()
                #print("p_flatten size is ,", p_flatten.size())
                #print("compressed msg, ", self._compressed_msg[name], 'rank, ', hvd.local_size())
                #print("hand is ", handle)
                for node_idx in range(hvd.size()):
                    if self._use_gpu:
                        p_flatten[self._compressed_msg[name][node_idx*msg_size*2 : \
                                node_idx*msg_size*2 + msg_size].type('torch.cuda.LongTensor')] += \
                                self._compressed_msg[name][node_idx*msg_size*2 + msg_size : \
                                node_idx*msg_size*2 + 2*msg_size]
                    else:
                        p_flatten[self._compressed_msg[name][node_idx*msg_size*2 : \
                                node_idx*msg_size*2 + msg_size].type('torch.LongTensor')] += \
                                self._compressed_msg[name][node_idx*msg_size*2 + msg_size : \
                                node_idx*msg_size*2 + 2*msg_size]

                p.grad.data = p.grad.data.view(g_size)
                if self._debug:
                    print("diff : ", torch.sum(self._v_ref[name] - p.grad.data))

        self._handles.clear()

    def step(self, closure=None):
        self.synchronize()
        return super(self.__class__, self).step(closure)


def DGCDistributedOptimizer(optimizer, named_parameters=None, use_gpu=True, momentum=0.9, weight_decay=1e-4):
    """
    An optimizer that wraps another torch.optim.Optimizer, 
    Compress gradients according to their magnitude
    using an allgather to reduce compressed gradient values before applying gradients to model weights.
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
               dict(_DGCOptimizer.__dict__))
    return cls(optimizer.param_groups, named_parameters,use_gpu, momentum, weight_decay)

