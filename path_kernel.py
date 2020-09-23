import torch
import copy

def compute_path_kernel_sum(model, dataloader, device):
    squared_copy = copy.deepcopy(model)

    for name, p in squared_copy.named_parameters():
        p.data = p.data ** 2

    data, _ = dataloader[0] # looking at just the input data shape to get dimensions for torch.ones (Synflow-like input)
    input_dim = list(data.shape)[1:]  # 1, 28, 28
    r_pk_ones = torch.ones([1] + input_dim).to(device)  # 1, 1, 28, 28

    squared_copy(r_pk_ones).sum().backward()

    path_kernel_sum = 0
    for name, p in squared_copy.named_parameters():
        curr_sum = torch.sum(torch.clone(p.grad).detach())
        print(name, curr_sum)
        path_kernel_sum += curr_grad_sq_sum

    return path_kernel_sum