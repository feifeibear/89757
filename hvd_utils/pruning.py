import torch
import numpy as np
from torch.autograd import Variable
from time import time

def struct_pruning(x, perc, offset):
    x_size = x.size()
    x_len = 1;
    for dim in x.size():
        x_len *= dim
    slice_size = int(x_len * perc)+1
    idx = []
    if offset + slice_size > x_len:
        tail = range(offset, x_len)
        offset = slice_size - (x_len - offset)
        head = range(0, offset)
        idx = [i for j in (head, tail) for i in j]
    else:
        idx = range(offset, offset + slice_size)

    if torch.cuda.is_available():
        mask = torch.zeros(x_len).cuda()
    else:
        mask = torch.zeros(x_len)
    mask[idx] = 1.0
    mask = mask.view(x_size)
    return mask


def prune_relative_perc(x, y, perc):
    x_size = x.size()
    x_len = 1;
    for dim in x.size():
        x_len *= dim
    x_flatten = torch.abs(x.view(x_len))
    y_flatten = torch.abs(y.view(x_len))
    # torch.save(x_flatten, './tensors/x_flatten.pt')
    top_k = int(x_len * perc) + 1
    #print(x)
    #print("x_flatten norm : ", x_flatten.norm())
    #print("x_len : ", x_len, "debug top_k : ", top_k)

    #if top_k < 1:
    #    top_k = 1
    _, x_top_idx = torch.topk(x_flatten / y_flatten, top_k, 0, True, largest=True)
    #x_top_idx,_ = torch.sort(x_top_idx)
    #print('nonzero indices are : ', x_top_idx)
    # torch.save(x_top_idx, './tensors/x_top_idx.pt')

    # for i in x_top_idx:
    #     if i >= x_len:
    #         print("Error in top_k", "idx : ", i, " len : ", x_len)
    # x_top_idx = torch.LongTensor([i for i in range(x_len)]).cuda()
    #print(x_top_val, x_top_idx)

    if torch.cuda.is_available():
        mask = torch.zeros(x_len).cuda()
    else:
        mask = torch.zeros(x_len)

    mask[x_top_idx] = 1.0
    mask = mask.view(x_size)
    return mask

def prune_bin(x, bin_size=1024, topk=1):
    r"""select upper(x/bin_size) elem from x"""
    x_size = x.size()
    x_len = 1;
    for dim in x.size():
        x_len *= dim

    if torch.cuda.is_available():
        mask = torch.zeros(x_len).cuda()
    else:
        mask = torch.zeros(x_len)

    x_flatten = torch.abs(x.view(x_len))
    offset = 0
    x_top_idx = []
    while offset < x_len:
        if offset + bin_size < x_len:
            _, idx = torch.topk(x_flatten[offset: offset + bin_size], topk, 0, largest=True, sorted=False)
            mask[idx + offset] = 1.0
        else:
            _, idx = torch.topk(x_flatten[offset:], min(topk, x_len - offset), 0, largest=True, sorted=False)
            mask[idx + offset] = 1.0

        offset += bin_size

    mask = mask.view(x_size)
    return mask

def select_top_k_v2(x, pruning_ratio, U, V):
    r"""a fast function to select top k% abs largest elements, and assign indices to mask"""
    x_size = x.size()
    x_len = 1;
    for dim in x.size():
        x_len *= dim
    top_k = int(x_len * pruning_ratio) + 1
    _, x_idx = torch.topk(torch.abs(x.view(x_len)), top_k, 0, largest=True, sorted=False)
#    x_bottom_val, x_bottom_idx = torch.topk((x.view(x_len)), top_k, 0, largest=False, sorted=False)
    #x_val = torch.cat((x_top_val, x_bottom_val))
    #x_idx = torch.cat((x_top_idx, x_bottom_idx))
    x_val = torch.index_select(x.view(x_len), 0, x_idx)
    U = U.view(-1)
    V = V.view(-1)
    U[x_idx] = 0.0
    V[x_idx] = 0.0
    U = U.view(x_size)
    V = V.view(x_size)
    return U, V, x_val, x_idx

def select_top_k_appr(x, pruning_ratio, mask):
    r"""a fast function to select top k% abs largest elements, and assign indices to mask"""
    x_size = x.size()
    x_len = 1;
    for dim in x.size():
        x_len *= dim
    top_k = int(x_len * pruning_ratio) + 1
    _, x_idx = torch.topk(torch.abs(x.view(x_len)), top_k, 0, largest=True, sorted=False)
    # x_bottom_val, x_bottom_idx = torch.topk((x.view(x_len)), top_k, 0, largest=False, sorted=False)
    # x_val = torch.cat((x_top_val, x_bottom_val))
    # x_idx = torch.cat((x_top_idx, x_bottom_idx))
    x_val = torch.index_select(x.view(x_len), 0, x_idx)
    mask = mask.view(-1)
    mask.zero_()
    mask[x_idx] = 1.0
    mask = 1.0 - mask
    mask = mask.view(x_size)
    return mask, x_val, x_idx

def select_top_k_thdv3(x, pruning_ratio, param = 0.0):
    r"""a fast function to select top k% abs largest elements, and assign indices to mask"""
    x_size = x.size()
    x_len = 1;
    for dim in x.size():
        x_len *= dim
    x_flatten = x.view(-1)
    x_abs = torch.abs(x_flatten)
    top_k = int(x_len * pruning_ratio) + 1
    max_val = torch.max(x_abs)
    mean_val = torch.mean(x_abs)
    #print("max_val ", max_val, " mean_val ", mean_val, " threshold ", threshold)

    # roughly select top
    rough_indices = []
    param = 0.3
    threshold = mean_val + param * (max_val - mean_val)
    x_sparse = x_abs > threshold
    rough_indices = torch.nonzero(x_sparse).view(-1)
    rough_val = torch.index_select(x_flatten, 0, rough_indices)

    #print(len(rough_indices), top_k, param, max_val, mean_val)
    # _, fine_indices = torch.topk(rough_val, top_k, 0, largest=True, sorted=False)
    # x_idx = torch.index_select(rough_indices, 0, fine_indices)

    # x_val = torch.index_select(x_flatten, 0, x_idx)
    return rough_val, rough_indices



def select_top_k_thdv2(x, pruning_ratio, param = 0.0):
    r"""a fast function to select top k% abs largest elements, and assign indices to mask"""
    x_size = x.size()
    x_len = 1;
    for dim in x.size():
        x_len *= dim
    x_flatten = x.view(-1)
    x_abs = torch.abs(x_flatten)
    top_k = int(x_len * pruning_ratio) + 1
    max_val = torch.max(x_abs)
    mean_val = torch.mean(x_abs)
    #print("max_val ", max_val, " mean_val ", mean_val, " threshold ", threshold)

    # roughly select top
    rough_indices = []
    param = 0.3
    threshold = 0.0
    while(len(rough_indices) < top_k):
        threshold = mean_val + param * (max_val - mean_val)
        x_sparse = x_abs > threshold
        rough_indices = torch.nonzero(x_sparse).view(-1)
        param -= 0.1
    rough_val = torch.index_select(x_flatten, 0, rough_indices)

    #print(len(rough_indices), top_k, param, max_val, mean_val)
    # _, fine_indices = torch.topk(rough_val, top_k, 0, largest=True, sorted=False)
    # x_idx = torch.index_select(rough_indices, 0, fine_indices)

    # x_val = torch.index_select(x_flatten, 0, x_idx)
    return rough_val, rough_indices




def select_top_k_thd(x, pruning_ratio, mask):
    r"""a fast function to select top k% abs largest elements, and assign indices to mask"""
    x_size = x.size()
    x_len = 1;
    for dim in x.size():
        x_len *= dim
    x_flatten = x.view(-1)
    top_k = int(x_len * pruning_ratio) + 1
    max_val = torch.max(torch.abs(x))
    mean_val = torch.mean(torch.abs(x))
    #print("max_val ", max_val, " mean_val ", mean_val, " threshold ", threshold)

    # roughly select top
    param = 0.5
    rough_indices = []
    while len(rough_indices) < top_k:
        threshold = mean_val + param * (max_val - mean_val)
        x_sparse = torch.abs(x_flatten) > threshold
        rough_indices = torch.nonzero(x_sparse).view(-1)
        param -= 0.1

    rough_val = torch.index_select(torch.abs(x_flatten), 0, rough_indices)

    #print(len(rough_indices), top_k)
    _, fine_indices = torch.topk(rough_val, top_k, 0, largest=True, sorted=False)
    x_idx = torch.index_select(rough_indices, 0, fine_indices)

    x_val = torch.index_select(x_flatten, 0, x_idx)

    mask = mask.view(-1)
    mask.zero_()
    mask[x_idx] = 1.0
    mask = 1.0 - mask
    mask = mask.view(x_size)
    return mask, x_val, x_idx


def select_top_k(x, pruning_ratio, mask):
    r"""a fast function to select top k% abs largest elements, and assign indices to mask"""
    x_size = x.size()
    x_len = 1;
    for dim in x.size():
        x_len *= dim
    top_k = int(x_len * pruning_ratio) + 1
    x_top_val, x_top_idx = torch.topk(torch.abs(x.view(x_len)), top_k, 0, largest=True, sorted=False)
    x = x.view(-1)
    for i in range(top_k):
        x_top_val[i] = x[x_top_idx[i]]
    mask = mask.view(-1)
    mask.zero_()
    mask[x_top_idx] = 1.0
    mask = mask.view(x_size)
    return mask, x_top_val, x_top_idx

def prune_perc(x, perc):
    r"""this is an old API"""
    x_size = x.size()
    x_len = 1;
    for dim in x.size():
        x_len *= dim
    x_flatten = torch.abs(x.view(x_len))
    # torch.save(x_flatten, './tensors/x_flatten.pt')
    top_k = int(x_len * perc) + 1

    _, x_top_idx = torch.topk(x_flatten, top_k, 0, largest=True, sorted=False)
    if torch.cuda.is_available():
        mask = torch.zeros(x_len).cuda()
    else:
        mask = torch.zeros(x_len)

    mask[x_top_idx] = 1.0
    mask = mask.view(x_size)
    return mask


def check_sparsity(x):
    nnz = len(torch.nonzero(x == 0.))
    x_len = 1;
    for dim in x.size():
        x_len *= dim
    return 1- nnz / x_len
    #print(mask)
#grad = torch.sparse.FloatTensor(x_top_idx, x_top_val, torch.Size([x_len]))#.to_dense().view(x_size)
#print(grad)
#residue = x - grad
#print(grad, residue)
def kth(arr, topk, sample_rate=1):
    # to numpy array
    arr = arr.cpu().numpy().ravel()

    if sample_rate < 1:
        arr = np.random.choice(arr, int(arr.size * sample_rate), replace=False)

    arr = np.abs(arr)
    num = arr.size

    k = int(max(1, topk * num))
    ids = np.argpartition(arr, -k)[-k:]
    thr = float(np.min(arr[ids]))

    return thr

def prune_perc_sample(x, perc):
    x_size = x.size()
    x_len = np.prod(x_size)
    thr = 0
    thr = kth(x, perc, 1.0)
    # if(x_len > 1e4):
    #     thr = kth(x, perc, 0.01)
    # else:
    #     thr = kth(x, perc, 1.0)
    mask = (x.abs() >= thr).type(x.type())
    return mask



if __name__ == '__main__':
    torch.manual_seed(123)
    #x = torch.randn(10, 10) #FloatTensor([[1, 2, 3], [4, 5, 6]])
    x = torch.randn(33278, 1500) #FloatTensor([[1, 2, 3], [4, 5, 6]])
    #x = torch.randn(100, 100) #FloatTensor([[1, 2, 3], [4, 5, 6]])
    #x = torch.randn(256, 256, 3, 3) #FloatTensor([[1, 2, 3], [4, 5, 6]])
    #x = torch.randn(14000000,) #FloatTensor([[1, 2, 3], [4, 5, 6]])
    x = x.cuda()
    x_flatten = x.view(-1)
    x_len = 1;
    for dim in x.size():
        x_len *= dim
    print("x_len : ", x_len)
    ratio = 0.001

    # start = time()
    # for i in range(100):
    #     mask = prune_bin(x, 1024, 1)
    # stop = time()
    # print(str(stop-start), "s")
    mask1 = torch.zeros(x_len).cuda()
    mask2 = torch.zeros(x_len).cuda()

    start = time()
    for i in range(100):
        mask1, val, idx = select_top_k_thd(x, ratio, mask1)
    torch.cuda.synchronize()
    stop = time()
    print("prune_perc function run time : ", str((stop-start)/100), "s")

    start = time()
    for i in range(100):
        mask2, _, idx= select_top_k_appr(x, ratio, mask2)
    torch.cuda.synchronize()
    stop = time()
    print("select_top_k_appr function run time : ", str((stop-start)/100), "s")

    print("Time transfer in 10Gps Ethernet : ", str(x_len * 8 / (1e9/8)), "s")
    diff = mask1 - mask2
    print("diff is, ", torch.sum(diff))

    start = time()
    for i in range(100):
        _, x_top_idx = torch.topk(x_flatten, int(x_len * ratio)+1, 0, largest=True, sorted=False)
    torch.cuda.synchronize()
    stop = time()
    print("top-k API run time : ", str((stop-start)/100), "s")

    start = time()
    for i in range(100):
        x_size = x.size()
        x_len = 1;
        for dim in x.size():
            x_len *= dim
        top_k = int(x_len * ratio) + 1
        _, x_idx = torch.topk((x.view(x_len)), top_k, 0, largest=True, sorted=False)
        #x_bottom_val, x_bottom_idx = torch.topk((x.view(x_len)), top_k, 0, largest=False, sorted=False)
        #x_val = torch.cat((x_top_val, x_bottom_val))
        #x_idx = torch.cat((x_top_idx, x_bottom_idx))
        x_val = torch.index_select(x.view(x_len), 0, x_idx)
        mask1 = mask1.view(-1)
        mask1.zero_()
        mask1[x_idx] = 1.0
        mask1 = mask1.view(x_size)
    torch.cuda.synchronize()
    stop = time()
    print("top-k + clear time : ", str((stop-start)/100), "s")




    # start = time()
    # for i in range(100):
    #     prune_perc_sample(x, 0.001)
    # stop = time()
    # print(str(stop-start), "s")

    # start = time()
    # for i in range(100):
    #     prune_perc(x, 0.001)
    # stop = time()
    # print(str(stop-start), "s")

    # thr = kth(x, 5, 1.0)
    # mask = (x.abs() >= thr).type(x.type())
    # nmask = (x.abs() < thr).type(x.type())
    # print(mask)
    # print(nmask)
    # mask = prune_perc(x, 0.2)
    # offset = 0
    # for i in range(20):
    #     mask = struct_pruning(x, 0.2, offset)
    #     x_len = np.prod(x.size())
    #     slice_size = int(x_len * 0.2) + 1
    #     if offset + slice_size > x_len:
    #         offset = slice_size - (x_len - offset)
    #     else:
    #         offset += slice_size
    #     print(mask)



    # print(x)
    # print(x * mask)
    # print(x * (1. - mask))

    # sx = (x * (1. - mask))
    # print("sparse ", check_sparsity(x))
    # print("sparse ", check_sparsity(x * mask))
