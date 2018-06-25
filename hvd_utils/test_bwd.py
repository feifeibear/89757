import torch
import time
import torch
import horovod.torch as hvd
import numpy as np
from horovod.torch.mpi_ops import poll, synchronize
from pruning import select_top_k_thdv3, select_top_k_appr, check_sparsity, prune_perc

if __name__ == '__main__':
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    torch.manual_seed(123)
    #p_g = torch.randn(256,256,3,3).cuda()
    #p_g = torch.randn(128,128,3,3).cuda()
    data_size = []
    bwd = []

    handles = {}
    for i in range(20):
        size =  128 * 2**i
        p_g = torch.randn(size).cuda()
        torch.cuda.synchronize()
        begin_time = time.time()
        rept = 10
        if i < 10:
            rept = 10000
        for i in range(rept):
            handle = hvd.allreduce_async_(p_g, average=False)
            handles[i] = handle
        for i in range(rept):
            synchronize(handles[i])
        torch.cuda.synchronize()
        end_time = time.time()
        if hvd.local_rank() == 0:
            print(size * 8 / 1024, "KB")
            #print('allreduce time, ', (end_time - begin_time) / rept)
            bandwidth = size*32 * rept /(end_time - begin_time) / 1e9
            print('allreduce bandwidth, ', bandwidth)
            data_size.append(size)
            bwd.append(bandwidth)
    print(data_size)
    print(bwd)
    exit(0)

    #torch.cuda.synchronize()
    #begin_time = time.time()
    #rept = 10
    #for i in range(rept):
    #    p_g = p_g.type('torch.cuda.LongTensor')
    #torch.cuda.synchronize()
    #end_time = time.time()
    #if hvd.local_rank() == 0:
    #    print('format transfer time, ', end_time - begin_time)

    torch.cuda.synchronize()
    begin_time = time.time()
    _, fine_indices = torch.topk(p_g, int(len(p_g) * 0.001), 0, largest=True, sorted=False)
    torch.cuda.synchronize()
    end_time = time.time()
    if hvd.local_rank() == 0:
        print('select 2D time, ', end_time - begin_time)


    torch.cuda.synchronize()
    begin_time = time.time()
    p_flatten = p_g.view(-1)
    p_flatten.zero_()
    p_flatten.mul_(p_flatten)
    torch.cuda.synchronize()
    end_time = time.time()
    if hvd.local_rank() == 0:
        print('flatten time, ', end_time - begin_time)
    #msg_size = int(torch.numel(p_g) * 0.001)
    torch.cuda.synchronize()
    begin_time = time.time()

    compressed_val, compressed_idx = select_top_k_thdv3(p_g, 0.001)
    torch.cuda.synchronize()
    end_time = time.time()
    if hvd.local_rank() == 0:
        print('select time, ', end_time - begin_time)


    compressed_msg = torch.cat([compressed_idx.type('torch.cuda.FloatTensor'), compressed_val])
    torch.cuda.synchronize()
    begin_time = time.time()
    msg_size = len(compressed_val)
    node_idx= 0
    a = compressed_msg[node_idx*msg_size*2 : \
            node_idx*msg_size*2 + msg_size].type('torch.cuda.LongTensor')
    torch.cuda.synchronize()
    end_time = time.time()
    if hvd.local_rank() == 0:
        print('type transfer time, ', end_time - begin_time)


    torch.cuda.synchronize()
    begin_time = time.time()
    for node_idx in range(4):
        p_flatten[compressed_msg[node_idx*msg_size*2 : \
        node_idx*msg_size*2 + msg_size].type('torch.cuda.LongTensor')] += \
        compressed_msg[node_idx*msg_size*2 + msg_size : \
        node_idx*msg_size*2 + 2*msg_size]

    torch.cuda.synchronize()
    end_time = time.time()
    print('decompress time, ', end_time - begin_time)
