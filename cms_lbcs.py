import math
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

# torch.manual_seed(3123)  # See https://arxiv.org/abs/2109.08203
torch.manual_seed(0)


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
        # head_ratios = F.normalize(head_ratios, p=1.0, dim=0)
        head_ratios = (F.normalize(head_ratios, p=1.0, dim=0) +
                       (0.001 / self.n_heads)) / 1.001
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


class CMS_LBCS_args:
    def __init__(self, terminate_loss=-1, multi_GPU=False) -> None:
        self.n_step = 500000
        self.terminate_loss = terminate_loss
        self.one_by_one = False
        self.random_init = False
        self.head_from_hamil = True
        self.rescale_head_grad = True
        self.normalize_grad = False
        self.multi_GPU = multi_GPU

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

    def set_init_heads(self, head_ratios, heads):
        self.head_ratios = head_ratios
        self.heads = heads
        self.head_from_args = True

def train_cms_lbcs(n_head, hamil, batch_size, args=None):

    if args == None:
        args = CMS_LBCS_args()

    n_qubit = hamil.n_qubit
    coeffs, pauli_tensor = hamil.get_one_hot_tensor()
    coeffs = torch.tensor(coeffs)
    pauli_tensor = torch.tensor(pauli_tensor)
    pauli_tensor = get_no_zero_pauliwords(pauli_tensor)
    pauli_tensor = pauli_tensor.to(device)
    coeffs = coeffs.to(device)
    n_pauliwords = len(coeffs)

    batch_size = n_pauliwords // math.ceil(n_pauliwords / batch_size) + 1

    if args.verbose:
        print("Hamiltonian contain {} terms".format(n_pauliwords))
        print("Acutal batch size: ", batch_size)

    if args.random_init:
        head_ratios = (5 + 1 * torch.rand((n_head,))).to(device)
        heads = (5 + 1 * torch.rand((n_head, n_qubit, 3))).to(device)

    if args.head_from_hamil:
        term_rank = torch.argsort(coeffs, descending=True)
        head_ratios = coeffs[term_rank][:n_head]
        heads = pauli_tensor[term_rank][:n_head]

    if args.head_from_args:
        print("Using initial value from args")
        head_ratios = torch.asarray(args.head_ratios)
        heads = torch.asarray(args.heads)
        assert len(head_ratios) == n_head or n_head == -1

    model = LargeLBCS(head_ratios, heads, freeze_heads = args.freeze_heads).to(device)

    for name, param in model.named_parameters():
        if name == "heads":
            heads_param = param
        elif name == "head_ratios":
            head_ratios_param = param

    lr = args.lr
    lr_ratio = lr * args.bilevel_ratio

    optimizer = torch.optim.Adam([{"params": [heads_param]}, {"params": [
                                 head_ratios_param], "lr": lr_ratio}], lr=lr, weight_decay=1e-5)

    n_epoch = 0
    batch_n = 0
    avg_loss_for_epoch = 0
    loss_in_epoch = []
    loss_of_epoches = []
    loss_at_steps = []
    min_total_loss = 999999999999
    stop_count = 0

    with tqdm(range(args.n_step), ncols=100) as pbar:
        for i_step in pbar:
            optimizer.zero_grad()
            if batch_n == 0:
                # Calculate the loss
                total_loss = avg_loss_for_epoch
                pbar.set_description(
                    'Var: {:.6f}, Epoch: {}'.format(total_loss, n_epoch))
                # Permute the pauliwords
                randperm = torch.randperm(n_pauliwords)
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
            params = model.named_parameters()
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
                avg_loss_for_epoch = float(sum(loss_in_epoch).detach().numpy())
                loss_at_steps.append((avg_loss_for_epoch, i_step))
                loss_of_epoches.append(avg_loss_for_epoch)
                total_loss = avg_loss_for_epoch
                loss_in_epoch = []
                # Log the run
                if args.logger is not None and n_epoch % 10 == 0:
                    data = {"loss_at_steps": loss_at_steps,
                            "loss_of_epoches": loss_of_epoches}
                    flag = args.logger(data)
                    if flag:
                        return heads.cpu().detach().numpy(), head_ratios.cpu().detach().numpy()
                # Determine whether to terminate by args.min_factor_for_decreasing_step or args.terminate_loss
                if min_total_loss - total_loss > total_loss * args.min_factor_for_decreasing_step:
                    stop_count = 0
                    min_total_loss = total_loss
                else:
                    stop_count += n_pauliwords / batch_size
                    if stop_count >= args.n_non_decreasing_step_to_stop:
                        print(
                            f"Loss is not decreasing by {args.min_factor_for_decreasing_step} for {args.n_non_decreasing_step_to_stop} steps")
                        print(f"Min loss: {min_total_loss}")
                        break
                if total_loss < args.terminate_loss:
                    break

    return head_ratios.cpu().detach().numpy(), heads.cpu().detach().numpy()


if __name__ == '__main__':
    from mizore.testing.hamil import get_test_hamil
    mol_name = "H2O_26_JW"
    n_head = 1000
    hamil, _ = get_test_hamil("mol", mol_name).remove_constant()
    train_cms_lbcs(n_head, hamil, 400, CMS_LBCS_args())