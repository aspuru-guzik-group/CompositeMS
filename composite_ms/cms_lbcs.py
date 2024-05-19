import math
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

# torch.manual_seed(3123)  # See https://arxiv.org/abs/2109.08203

dtype = torch.float32
p = 2


def project_parameter(model, p):
    with torch.no_grad():
        model.heads = nn.Parameter(F.normalize(model.heads, p=p, dim=-1))
        model.head_ratios = nn.Parameter(F.normalize(model.head_ratios, p=p, dim=0))
        #model.head_ratios = nn.Parameter(F.relu(model.head_ratios))
        #model.heads = nn.Parameter(F.relu(model.heads))
        model.head_ratios = nn.Parameter(abs(model.head_ratios))
        model.heads = nn.Parameter(abs(model.heads))
        model.head_ratios += (0.001 / len(model.heads)) / 1.001
        model.heads += (0.001 / len(model.heads)) / 1.001
        model.head_ratios = nn.Parameter(F.normalize(model.head_ratios, p=p, dim=0))
        model.heads = nn.Parameter(F.normalize(model.heads, p=p, dim=-1))


class LargeLBCS(nn.Module):
    def __init__(self, init_head_ratios, init_heads, freeze_heads=False):
        super(LargeLBCS, self).__init__()
        self.activator = torch.nn.Softplus()
        head_ratios = init_head_ratios
        heads = init_heads
        heads = F.normalize(heads, p=1.0, dim=-1)
        head_ratios = F.normalize(head_ratios, p=1.0, dim=0)

        self.p = p
        self.heads = torch.nn.Parameter(heads ** (1 / p),
                                        requires_grad=(not freeze_heads))
        self.head_ratios = torch.nn.Parameter(head_ratios ** (1 / p), requires_grad=True)
        self.n_heads = len(heads)
        project_parameter(self, p)

        # print(sum(head_ratios))
        # print()

    def get_heads_and_heads_ratio(self):
        heads = self.heads ** self.p
        head_ratios = self.head_ratios ** self.p
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
    # coeffs = torch.tensor(coeffs,)
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

        self.lr = 1e-3
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

    torch.manual_seed(0)

    n_qubit = hamil.n_qubit
    coeffs, pauli_tensor = hamil.get_one_hot_tensor()
    coeffs = torch.tensor(coeffs, dtype=dtype)
    pauli_tensor = torch.tensor(pauli_tensor, dtype=dtype)
    pauli_tensor = get_no_zero_pauliwords(pauli_tensor)
    pauli_tensor = pauli_tensor.to(device)
    coeffs = coeffs.to(device)
    n_pauliwords = len(coeffs)

    batch_size = n_pauliwords // math.ceil(n_pauliwords / batch_size) + 1

    if args.verbose:
        print("Hamiltonian contain {} terms".format(n_pauliwords))
        print("Acutal batch size: ", batch_size)

    if args.random_init:
        head_ratios = (1 + 1 * torch.rand((n_head,), dtype=dtype)).to(device)
        heads = (1 + 1 * torch.rand((n_head, n_qubit, 3), dtype=dtype)).to(device)

    if args.head_from_hamil:
        term_rank = torch.argsort(coeffs, descending=True)
        head_ratios = coeffs[term_rank][:n_head]
        head_ratios += torch.ones((n_head,), dtype=dtype) * 0.01
        heads = pauli_tensor[term_rank][:n_head]
        heads += torch.ones((n_head, n_qubit, 3), dtype=dtype) * 0.01

    if args.head_from_args:
        print("Using initial value from args")
        head_ratios = torch.asarray(args.head_ratios)
        heads = torch.asarray(args.heads)
        assert len(head_ratios) == n_head or n_head == -1

    model = LargeLBCS(head_ratios, heads, freeze_heads=args.freeze_heads).to(device)

    for name, param in model.named_parameters():
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

    with tqdm(range(args.n_step), ncols=100) as pbar:
        for i_step in pbar:

            if args.alternate_training_ratio > 0:
                divider = i_step % args.alternate_training_n_steps
                if divider < args.alternate_training_ratio * args.alternate_training_n_steps:
                    model.heads.requires_grad = True
                    model.head_ratios.requires_grad = False
                else:
                    model.heads.requires_grad = False
                    model.head_ratios.requires_grad = True

            optimizer.zero_grad()
            # Manually zero the gradients
            # model.heads.grad = None
            # model.head_ratios.grad = None

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
            # optimizer.step()
            # project_parameter(model, p)

            # Manually update the parameters
            with torch.no_grad():
                # print the norm of the gradient
                # if torch.norm(head_ratios_grad) > torch.norm(model.head_ratios) * 10:
                #    head_ratios_grad = head_ratios_grad / torch.norm(head_ratios_grad) * torch.norm(#model.head_ratios) * 10
                # print(torch.norm(head_ratios_grad) , torch.norm(model.head_ratios))
                model.head_ratios -= lr_for_ratios * (1+i_step)**(-0.9) * head_ratios_grad
                # remove the negative values and set them to 0
                model.heads -= lr * (1+i_step)**(-0.95) * heads_grad
                project_parameter(model, p)

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
    from composite_ms.hamil import get_test_hamil

    mol_name = "H2O_26_JW"
    n_head = 1000
    hamil, _ = get_test_hamil("mol", mol_name).remove_constant()
    train_cms_lbcs(n_head, hamil, 400, CMS_LBCS_args())
