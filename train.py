import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from path_kernel import compute_path_kernel_sum

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10,
          save_dir="", compute_init_outputs=False, compute_init_grads=False):
    model.train()
    total = 0
    batch_output, batch_target = None, None
    save_init_dir = save_dir + "/init"
    if not os.path.exists(save_init_dir):
        os.makedirs(save_init_dir)
    print("In train, save_init_dir is:", save_init_dir)
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        if (compute_init_outputs and epoch == 0):
            optimizer.zero_grad()
            output = model(data)
            with open(save_init_dir + f"/init_output_{epoch}_{batch_idx}.npy", 'wb') as f:
                np.save(f, output.cpu().data)
        if (compute_init_grads and epoch == 0):
            optimizer.zero_grad()
            output = model(data)
            output.sum().backward()
            for name, p in model.named_parameters():
                p_grad = torch.flatten(torch.clone(p.grad).detach())
                p_data = torch.flatten(torch.clone(p.data).detach())
                with open(save_init_dir + f"/init_grad_{epoch}_{name}_{batch_idx}.npy", 'wb') as f:
                    np.save(f, p_grad.cpu().data)
                with open(save_init_dir + f"/init_param_{epoch}_{name}_{batch_idx}.npy", 'wb') as f:
                    np.save(f, p_data.cpu().data)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        if batch_idx == 0:
            batch_output, batch_target = output, target 
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset), batch_output, batch_target

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5

def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs,
                    verbose, compute_path_kernel = False, track_weight_movement = False,
                    save_dir = "", compute_init_path_kernel=False, save_init_path_kernel_output_path="", init_path_kernel_row_name = "",
                    compute_init_outputs=False, compute_init_grads=False):
    path_kernel = 0

    if compute_init_path_kernel:
        path_kernel = compute_path_kernel_sum(model, train_loader, device)
        with open(save_init_path_kernel_output_path, 'a') as f:
            f.write(f"{init_path_kernel_row_name}, {str(path_kernel)}\n")

    allw0 = -1
    weight_movement_norm = 0

    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5, path_kernel, weight_movement_norm]]

    for epoch in tqdm(range(epochs)):
        train_loss, batch_output, batch_target = train(model, loss, optimizer, train_loader, device, epoch, verbose,
                                                       save_dir=save_dir, compute_init_outputs=compute_init_outputs,
                                                       compute_init_grads=compute_init_grads)

        # save batch_output, batch_target:
        # print(save_dir, type(batch_output), type(batch_target)) 
        with open(save_dir + f"/{epoch}_output.npy", 'wb') as f:
            np.save(f, batch_output.cpu().data)

        with open(save_dir + f"/{epoch}_target.npy", 'wb') as f:
            np.save(f, batch_target.cpu().data)

        # compute path kernel
        if compute_path_kernel:
            path_kernel = compute_path_kernel_sum(model, train_loader, device)

        # compute weights movement
        if track_weight_movement:
            if epoch == 0:
                allw0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy().copy()
                weight_movement_norm = np.linalg.norm(allw0)
            else:
                allw = torch.nn.utils.parameters_to_vector(model.parameters()).detach().cpu().numpy().copy()
                weight_movement_norm = np.linalg.norm(allw - allw0)

        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        row = [train_loss, test_loss, accuracy1, accuracy5, path_kernel, weight_movement_norm]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy', 'path_kernel', 'weight_movement_norm']
    return pd.DataFrame(rows, columns=columns)


