import torch
import time
from pruning import select_top_k_thdv3, select_top_k_appr, check_sparsity, prune_perc

if __name__ == '__main__':
    torch.manual_seed(123)
    #p_g = torch.randn(256,256,3,3).cuda()
    p_g = torch.randn(128,128,3,3).cuda()
    g_size = p_g.size()

    torch.cuda.synchronize()
    begin_time = time.time()
    p_flatten = p_g.view(-1)
    p_flatten.zero_()
    p_flatten.mul_(p_flatten)
    torch.cuda.synchronize()
    end_time = time.time()
    print('flatten time, ', end_time - begin_time)
    #msg_size = int(torch.numel(p_g) * 0.001)
    torch.cuda.synchronize()
    begin_time = time.time()
    compressed_val, compressed_idx = select_top_k_thdv3(p_g, 0.001)
    torch.cuda.synchronize()
    end_time = time.time()
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
