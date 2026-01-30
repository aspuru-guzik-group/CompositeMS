import math
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

device = torch.device('cpu')
n_gpus = 0
if torch.cuda.is_available():
    device = torch.device('cuda')
    n_gpus = torch.cuda.device_count()
    print(f"Using CUDA GPU - {n_gpus} GPU(s) available")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    n_gpus = 1
    print("Using Apple Silicon GPU (MPS)")
else:
    print("Using CPU")

# torch.manual_seed(3123)  # See https://arxiv.org/abs/2109.08203

dtype = torch.float32


def setup_ddp(rank, world_size):
    """
    Initialize the distributed environment.
    
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_ddp_config():
    """
    Get DDP configuration from environment variables.
    Returns rank, local_rank, and world_size.
    """
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, local_rank, world_size

class LargeLBCS(nn.Module):
    def __init__(self, init_head_ratios, init_heads, freeze_heads = False):
        super(LargeLBCS, self).__init__()
        self.activator = torch.nn.Softplus()
        head_ratios = init_head_ratios
        heads = init_heads
        self.heads = torch.nn.Parameter(heads, requires_grad=(not freeze_heads))
        self.head_ratios = torch.nn.Parameter(head_ratios, requires_grad=True)
        self.n_heads = len(heads)

    def get_heads_and_heads_ratio(self):
        heads = self.activator(self.heads * 20)
        heads = F.normalize(heads, p=1.0, dim=-1)
        head_ratios = self.activator(self.head_ratios * 20)
        # This is term is for keeping sub-scheme unfrozen
        head_ratios += (0.001 / self.n_heads) / 1.001
        head_ratios = F.normalize(head_ratios, p=1.0, dim=0)
        return heads, head_ratios

    def forward(self, batch_pauli_tensor, batch_coeff):
        heads, head_ratios = self.get_heads_and_heads_ratio()
        loss_val = loss(heads, head_ratios, batch_pauli_tensor, batch_coeff)
        return loss_val, head_ratios, heads


def get_no_zero_pauliwords(pauliwords):
    anti_qubit_mask = 1.0 - torch.sum(pauliwords, dim=-1)
    anti_qubit_mask: torch.tensor = anti_qubit_mask.unsqueeze(2)
    anti_qubit_mask = anti_qubit_mask.repeat(1, 1, 3)
    no_zero_pauliwords = pauliwords + anti_qubit_mask
    return no_zero_pauliwords


def get_shadow_coverage(heads, no_zero_pauliwords, head_ratios):
    shadow_coverage = torch.einsum(
        "nqp, sqp -> nsq", no_zero_pauliwords, heads)
    coverage = torch.prod(shadow_coverage, dim=-1)
    coverage = torch.einsum("s, ns -> n", head_ratios, coverage)
    return coverage


# This loss is not average variance
def loss(heads, head_ratios, no_zero_pauliwords, coeffs):
    coverage = get_shadow_coverage(heads, no_zero_pauliwords, head_ratios)
    var = torch.sum(1.0 / coverage * (coeffs ** 2))
    return var

def get_variance_by_batches(heads, head_ratios, no_zero_pauliwords, coeffs, batch_size):
    loss_list = []
    batch_n = 0
    #coeffs = torch.tensor(coeffs,)
    while True:
        batch_pauli_tensor = no_zero_pauliwords[batch_n:batch_n + batch_size]
        batch_coeffs = coeffs[batch_n:batch_n + batch_size]
        loss_val = loss(heads, head_ratios, batch_pauli_tensor, batch_coeffs)
        loss_list.append(loss_val.detach().cpu())
        if batch_n + batch_size >= len(no_zero_pauliwords):
            break
        batch_n += batch_size
    return sum(loss_list)


class CMS_LBCS_args:
    """
    Arguments for CMS-LBCS training.
    
    Parameters:
        terminate_loss: Stop training when loss reaches this value (default: -1, disabled)
        multi_GPU: Enable multi-GPU training using DataParallel (default: False)
                   When True, automatically uses all available CUDA GPUs
    """
    def __init__(self, terminate_loss=-1, multi_GPU=False) -> None:
        self.n_step = 500000
        self.terminate_loss = terminate_loss
        self.one_by_one = False
        self.random_init = False
        self.head_from_hamil = True
        self.rescale_head_grad = True
        self.normalize_grad = False
        self.multi_GPU = multi_GPU  # Enable multi-GPU training if multiple GPUs available

        self.n_non_decreasing_step_to_stop = 1000
        self.min_factor_for_decreasing_step = 1e-3

        self.logger = None

        self.lr = 0.005
        self.bilevel_ratio = 0.1

        self.head_from_args = False
        self.head_ratios = None
        self.heads = None

        self.verbose = True

        self.freeze_heads = False


        self.alternate_training_ratio = -1
        self.alternate_training_n_steps = -1

        self.ratio_adam_beta_1 = 0.9

    def set_init_heads(self, head_ratios, heads):
        self.head_ratios = head_ratios
        self.heads = heads
        self.head_from_args = True



def train_cms_lbcs(n_head, hamil, batch_size, args=None):

    if args == None:
        args = CMS_LBCS_args()

    # Setup DDP if multi_GPU is enabled
    rank, local_rank, world_size = 0, 0, 1
    is_ddp = False
    
    if args.multi_GPU and n_gpus > 1:
        rank, local_rank, world_size = get_ddp_config()
        is_ddp = True
        
        # Only initialize if not already initialized
        if not dist.is_initialized():
            setup_ddp(rank, world_size)
        
        # Set device to local rank
        current_device = torch.device(f'cuda:{local_rank}')
        
        if rank == 0:
            print(f"Using DistributedDataParallel with {world_size} GPUs")
    else:
        current_device = device
        if args.multi_GPU and n_gpus <= 1:
            if rank == 0:
                print("Warning: multi_GPU requested but only 1 or fewer GPUs available. Using single GPU/CPU.")

    torch.manual_seed(0)

    n_qubit = hamil.n_qubit
    coeffs, pauli_tensor = hamil.get_one_hot_tensor()
    coeffs = torch.tensor(coeffs, dtype=dtype)
    pauli_tensor = torch.tensor(pauli_tensor, dtype=dtype)
    pauli_tensor = get_no_zero_pauliwords(pauli_tensor)
    pauli_tensor = pauli_tensor.to(current_device)
    coeffs = coeffs.to(current_device)
    n_pauliwords = len(coeffs)

    batch_size = n_pauliwords // math.ceil(n_pauliwords / batch_size) + 1

    if args.verbose and rank == 0:
        print("Hamiltonian contain {} terms".format(n_pauliwords))
        print("Acutal batch size: ", batch_size)

    if args.random_init:
        head_ratios = (1 + 1 * torch.rand((n_head,), dtype=dtype)).to(current_device)
        heads = (1 + 1 * torch.rand((n_head, n_qubit, 3), dtype=dtype)).to(current_device)

    if args.head_from_hamil:
        term_rank = torch.argsort(coeffs, descending=True)
        head_ratios = coeffs[term_rank][:n_head]
        heads = pauli_tensor[term_rank][:n_head]

    if args.head_from_args:
        if rank == 0:
            print("Using initial value from args")
        head_ratios = torch.asarray(args.head_ratios)
        heads = torch.asarray(args.heads)
        assert len(head_ratios) == n_head or n_head == -1

    model = LargeLBCS(head_ratios, heads, freeze_heads = args.freeze_heads).to(current_device)
    
    # Enable DDP if multi_GPU is enabled
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    # Access parameters (handle DDP wrapper)
    model_params = model.module if isinstance(model, DDP) else model
    for name, param in model_params.named_parameters():
        if name == "heads":
            heads_param = param
        elif name == "head_ratios":
            head_ratios_param = param

    lr = args.lr
    lr_for_ratios = lr * args.bilevel_ratio

    optimizer = torch.optim.Adam([{"params": [heads_param], "lr": lr}, {"params": [
                                 head_ratios_param], "lr": lr_for_ratios}], weight_decay=1e-5)

    n_epoch = 0
    batch_n = 0
    avg_loss_for_epoch = 0
    loss_in_epoch = []
    loss_of_epoches = []
    loss_at_steps = []
    min_total_loss = 999999999999
    stop_count = 0

    # Only show progress bar on rank 0
    pbar = tqdm(range(args.n_step), ncols=100, disable=(rank != 0))
    
    try:
        for i_step in pbar:

            if args.alternate_training_ratio > 0:
                divider = i_step % args.alternate_training_n_steps
                model_params = model.module if isinstance(model, DDP) else model
                if divider < args.alternate_training_ratio * args.alternate_training_n_steps:
                    model_params.heads.requires_grad = True
                    model_params.head_ratios.requires_grad = False
                else:
                    model_params.heads.requires_grad = False
                    model_params.head_ratios.requires_grad = True

            optimizer.zero_grad()
            if batch_n == 0:
                # Calculate the loss
                total_loss = avg_loss_for_epoch
                if rank == 0:
                    pbar.set_description(
                        'Var: {:.6f}, Epoch: {}'.format(total_loss, n_epoch))
                # Permute the pauliwords (use same random seed across all ranks for DDP)
                if is_ddp:
                    # Ensure all ranks use the same permutation
                    torch.manual_seed(n_epoch)
                randperm = torch.randperm(n_pauliwords, device=current_device)
                pauli_tensor = pauli_tensor[randperm, :]
                coeffs = coeffs[randperm]

            # Generate a batch
            batch_pauli_tensor = pauli_tensor[batch_n:batch_n + batch_size]
            batch_coeffs = coeffs[batch_n:batch_n + batch_size]

            # Pass forward
            loss_val, head_ratios, heads = model(
                batch_pauli_tensor, batch_coeffs)
            loss_val = torch.sum(loss_val)
            loss_in_epoch.append(loss_val.cpu())
            loss_val.backward()

            # Modify the gradient (rescale)
            model_params = model.module if isinstance(model, DDP) else model
            params = model_params.named_parameters()
            for name, param in params:
                if name == "heads":
                    heads_grad = param.grad
                elif name == "head_ratios":
                    head_ratios_grad = param.grad

            if heads_grad != None:
                if args.rescale_head_grad:
                    heads_grad /= head_ratios.unsqueeze(-1).unsqueeze(-1)
                
                # This is for experiment
                if args.normalize_grad:
                    heads_grad -= (torch.sum(heads_grad, dim=-1) / 3).unsqueeze(-1)
                    head_ratios_grad -= (torch.sum(head_ratios_grad,
                                            dim=-1) / (n_head))

            # Update the parameters
            optimizer.step()

            batch_n += batch_size
            if batch_n >= n_pauliwords:
                batch_n = 0
                n_epoch += 1
                avg_loss_for_epoch = float(sum(loss_in_epoch).detach().cpu().numpy())
                
                # Synchronize loss across all ranks if using DDP
                if is_ddp:
                    loss_tensor = torch.tensor(avg_loss_for_epoch, device=current_device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    avg_loss_for_epoch = float(loss_tensor.cpu().numpy() / world_size)
                
                loss_at_steps.append((avg_loss_for_epoch, i_step))
                loss_of_epoches.append(avg_loss_for_epoch)
                total_loss = avg_loss_for_epoch
                loss_in_epoch = []
                
                # Log the run (only on rank 0)
                if rank == 0 and args.logger is not None and n_epoch % 10 == 0:
                    data = {"loss_at_steps": loss_at_steps,
                            "loss_of_epoches": loss_of_epoches}
                    flag = args.logger(data)
                    if flag:
                        model_params = model.module if isinstance(model, DDP) else model
                        result = (model_params.heads.cpu().detach().numpy(), 
                                 model_params.head_ratios.cpu().detach().numpy())
                        if is_ddp:
                            cleanup_ddp()
                        return result
                
                # Determine whether to terminate
                if min_total_loss - total_loss > total_loss * args.min_factor_for_decreasing_step:
                    stop_count = 0
                    min_total_loss = total_loss
                else:
                    stop_count += n_pauliwords / batch_size
                    if stop_count >= args.n_non_decreasing_step_to_stop:
                        if rank == 0:
                            print(
                                f"Loss is not decreasing by {args.min_factor_for_decreasing_step} for {args.n_non_decreasing_step_to_stop} steps")
                            print(f"Min loss: {min_total_loss}")
                        break
                if total_loss < args.terminate_loss:
                    break
    
    finally:
        pbar.close()

    # Get final results from the model
    model_params = model.module if isinstance(model, DDP) else model
    head_ratios_result = model_params.head_ratios.cpu().detach().numpy()
    heads_result = model_params.heads.cpu().detach().numpy()
    
    # Cleanup DDP
    if is_ddp:
        cleanup_ddp()
    
    return head_ratios_result, heads_result


if __name__ == '__main__':
    from composite_ms.hamil import get_test_hamil
    mol_name = "H2O_26_JW"
    n_head = 1000
    hamil, _ = get_test_hamil("mol", mol_name).remove_constant()
    
    # Example: Enable multi-GPU training by setting multi_GPU=True
    # To run with DistributedDataParallel, use:
    # torchrun --nproc_per_node=NUM_GPUS composite_ms/cms_lbcs.py
    # or
    # python -m torch.distributed.launch --nproc_per_node=NUM_GPUS composite_ms/cms_lbcs.py
    
    args = CMS_LBCS_args(multi_GPU=False)
    train_cms_lbcs(n_head, hamil, 400, args)