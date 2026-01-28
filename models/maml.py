import os
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from    copy import deepcopy
import pickle

# Red atenuadora que genera, para cada grupo, un factor gamma (entre 0 y 1)
class Attenuator(nn.Module):
    def __init__(self, num_groups):
        super(Attenuator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_groups, num_groups),
            nn.ReLU(inplace=True),
            nn.Linear(num_groups, num_groups),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Función helper para agrupar parámetros por capa
def group_params_by_layer(model, groups_order):
    grouped = {group: [] for group in groups_order}
    for name, param in model.named_parameters():
        for group in groups_order:
            if name.startswith(group):
                grouped[group].append((name, param))
                break
    return grouped

class MAML(nn.Module):
    """
    MAML Learner
    """
    def __init__(self, args, learner, rep=0, writer=None):
        """

        :param args:
        """
        super(MAML, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.grad_clip = args.grad_clip
        self.second_order = args.second_order
        self.output_dir = args.output
        os.makedirs(os.path.join(self.output_dir, "weights"), exist_ok=True)
        self.rep = rep
        self.writer = writer

        self.net = learner
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter
    
    def fast_weights(self, loss, params, lr):
        grad = torch.autograd.grad(loss, params)
        if self.grad_clip is not None:
            grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
        params = list(map(lambda p: p[1] - lr * p[0], zip(grad, params)))
        return params

    def forward(self, dataset, save_weights=False, task_subset=None):
        task_num = len(dataset)
        suppsz, querysz = dataset.supp_sz, dataset.query_sz

        if task_subset is None:
            iterate_over = range(task_num)
        else:
            iterate_over = task_subset

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        maes = [0 for _ in range(self.update_step + 1)]
        mses = [0 for _ in range(self.update_step + 1)]
        rmse = [0 for _ in range(self.update_step + 1)]
        r2s = [0 for _ in range(self.update_step + 1)]


        for i in iterate_over:
            task = dataset[i]
            sampled_data = task.sample(suppsz, querysz)
            x_spt = sampled_data['support_features']
            y_spt = sampled_data['support_targets']
            x_qry = sampled_data['query_features']
            y_qry = sampled_data['query_targets']

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt, vars=None, bn_training=True)
            loss = F.mse_loss(logits, y_spt)
            grad = torch.autograd.grad(loss, self.net.parameters(), retain_graph=self.second_order, create_graph=self.second_order)
            if self.grad_clip is not None:
                grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            # fast_weights = self.fast_weights(loss, self.net.parameters(), self.update_lr)

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry, list(self.net.parameters()), bn_training=True)
                loss_q = F.mse_loss(logits_q, y_qry)
                losses_q[0] += loss_q

                mae_loss = torch.abs(logits_q - y_qry).mean().detach().cpu()
                maes[0] = maes[0] + mae_loss
                mses[0] = mses[0] + loss_q.item()
                rmse[0] = rmse[0] + torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
                r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
                r2s[0] = r2s[0] + r2

            # this is the loss and accuracy after the first update
            # [setsz, nway]
            logits_q = self.net(x_qry, fast_weights, bn_training=True)
            loss_q = F.mse_loss(logits_q, y_qry)
            losses_q[1] += loss_q
            with torch.no_grad():                
                # [setsz]
                mae_loss = torch.abs(logits_q - y_qry).mean().detach().cpu()
                maes[1] = maes[1] + mae_loss
                mses[1] = mses[1] + loss_q.item()
                rmse[1] = rmse[1] + torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
                r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
                r2s[1] = r2s[1] + r2

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt, fast_weights, bn_training=True)
                loss = F.mse_loss(logits, y_spt)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=self.second_order, create_graph=self.second_order)
                if self.grad_clip is not None:
                    grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                # fast_weights = self.fast_weights(loss, fast_weights, self.update_lr)

                logits_q = self.net(x_qry, fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.mse_loss(logits_q, y_qry)
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    mae_loss = torch.abs(logits_q - y_qry).mean().detach().cpu()
                    maes[k + 1] = maes[k + 1] + mae_loss
                    mses[k + 1] = mses[k + 1] + loss_q.item()
                    rmse[k + 1] = rmse[k + 1] + torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
                    r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
                    r2s[k + 1] = r2s[k + 1] + r2

            # Save adapted weights
            if save_weights:
                pickle.dump(fast_weights, open(os.path.join(self.output_dir, "weights", f'adapted_weights_{self.rep}_{task.name}.pkl'), 'wb'))

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # from torchviz import make_dot
        # dot = make_dot(loss_q, params=dict(self.net.named_parameters()))
        # dot.format = 'png'
        # dot.render('computational_graph')

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        if self.grad_clip is not None:
            for p in self.net.parameters():
                if p.grad is not None:
                    p.grad = torch.clamp_(p.grad, -self.grad_clip, self.grad_clip)
                else:
                    p.grad = torch.zeros_like(p)
        self.meta_optim.step()

        # mses = np.array(mses) / (querysz * task_num)
        # maes = np.array(maes) / (querysz * task_num)

        mses = np.array(mses) / task_num
        maes = np.array(maes) / task_num
        rmse = np.array(rmse) / task_num
        r2s = np.array(r2s) / task_num

        return mses, maes, rmse, r2s


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 3

        querysz = x_qry.size(0)

        maes = [0 for _ in range(self.update_step_test + 1)]
        mses = [0 for _ in range(self.update_step_test + 1)]
        rmses = [0 for _ in range(self.update_step_test + 1)]
        r2s = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.mse_loss(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters(), retain_graph=self.second_order, create_graph=self.second_order)
        if self.grad_clip is not None:
            grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        # fast_weights = self.fast_weights(loss, net.parameters(), self.update_lr)

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, list(net.parameters()), bn_training=True)
            # [setsz]
            mse_loss = F.mse_loss(logits_q, y_qry).item()
            mses[0] = mses[0] + mse_loss
            mae = torch.abs(logits_q - y_qry).mean().detach().cpu()
            maes[0] = maes[0] + mae
            rmse = torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
            rmses[0] = rmses[0] + rmse
            r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
            r2s[0] = r2s[0] + r2

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            mse_loss = F.mse_loss(logits_q, y_qry).item()
            mses[1] = mses[1] + mse_loss
            mae = torch.abs(logits_q - y_qry).mean().detach().cpu()
            maes[1] = maes[1] + mae
            rmse = torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
            rmses[1] = rmses[1] + rmse
            r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
            r2s[1] = r2s[1] + r2

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.mse_loss(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights, retain_graph=self.second_order, create_graph=self.second_order)
            if self.grad_clip is not None:
                grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            # fast_weights = self.fast_weights(loss, fast_weights, self.update_lr)

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.mse_loss(logits_q, y_qry)

            with torch.no_grad():
                mse_loss = loss_q.item()
                mses[k+1] = mses[k+1] + mse_loss
                mae = torch.abs(logits_q - y_qry).mean().detach().cpu()
                maes[k+1] = maes[k+1] + mae
                rmse = torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
                rmses[k+1] = rmses[k+1] + rmse
                r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
                r2s[k+1] = r2s[k+1] + r2


        # del net

        mses = np.array(mses)
        maes = np.array(maes)
        rmses = np.array(rmses)
        r2s = np.array(r2s)

        return net, fast_weights, mses, maes, rmses, r2s

    def finetuning_batched(self, x_spt, y_spt, x_qry, y_qry, batch_size, writer=None, task_name=None, update_steps=None, fast_weights=None):
        """
        Finetuning with minibatches
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :param batch_size: size of each minibatch
        :return:
        """
        assert len(x_spt.shape) == 3

        querysz = x_qry.size(0)

        if update_steps is None:
            update_steps = self.update_step_test

        maes = [0 for _ in range(update_steps + 1)]
        mses = [0 for _ in range(update_steps + 1)]
        rmses = [0 for _ in range(update_steps + 1)]
        r2s = [0 for _ in range(update_steps + 1)]

        net = deepcopy(self.net)

        def get_batches(x, y, batch_size):
            for i in range(0, len(x), batch_size):
                yield x[i:i + batch_size], y[i:i + batch_size]

        # 1. run the i-th task and compute loss for k=0
        if fast_weights is None:
            fast_weights = list(net.parameters())
        for x_spt_batch, y_spt_batch in get_batches(x_spt, y_spt, batch_size):
            logits = net(x_spt_batch, fast_weights, bn_training=True)
            loss = F.mse_loss(logits, y_spt_batch)
            grad = torch.autograd.grad(loss, fast_weights, retain_graph=False, create_graph=False)
            if self.grad_clip is not None:
                grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, list(net.parameters()), bn_training=True)
            # [setsz]
            mse_loss = F.mse_loss(logits_q, y_qry).item()
            mses[0] = mses[0] + mse_loss
            mae = torch.abs(logits_q - y_qry).mean().detach().cpu()
            maes[0] = maes[0] + mae
            rmse = torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
            rmses[0] = rmses[0] + rmse
            r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
            r2s[0] = r2s[0] + r2
            if writer is not None:
                writer.add_scalar(f'Adaptation/{task_name}/MSE', mse_loss, 0)
                writer.add_scalar(f'Adaptation/{task_name}/MAE', mae, 0)
                writer.add_scalar(f'Adaptation/{task_name}/RMSE', rmse, 0)
                writer.add_scalar(f'Adaptation/{task_name}/R2', r2, 0)


        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            mse_loss = F.mse_loss(logits_q, y_qry).item()
            mses[1] = mses[1] + mse_loss
            mae = torch.abs(logits_q - y_qry).mean().detach().cpu()
            maes[1] = maes[1] + mae
            rmse = torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
            rmses[1] = rmses[1] + rmse
            r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
            r2s[1] = r2s[1] + r2
            if writer is not None:
                writer.add_scalar(f'Adaptation/{task_name}/MSE', mse_loss, 1)
                writer.add_scalar(f'Adaptation/{task_name}/MAE', mae, 1)
                writer.add_scalar(f'Adaptation/{task_name}/RMSE', rmse, 1)
                writer.add_scalar(f'Adaptation/{task_name}/R2', r2, 1)

        for k in range(1, update_steps):
            for x_spt_batch, y_spt_batch in get_batches(x_spt, y_spt, batch_size):
                logits = net(x_spt_batch, fast_weights, bn_training=True)
                loss = F.mse_loss(logits, y_spt_batch)
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=False, create_graph=False)
                if self.grad_clip is not None:
                    grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.mse_loss(logits_q, y_qry)

            with torch.no_grad():
                mse_loss = loss_q.item()
                mses[k+1] = mses[k+1] + mse_loss
                mae = torch.abs(logits_q - y_qry).mean().detach().cpu()
                maes[k+1] = maes[k+1] + mae
                rmse = torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
                rmses[k+1] = rmses[k+1] + rmse
                r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
                r2s[k+1] = r2s[k+1] + r2
                if writer is not None:
                    writer.add_scalar(f'Adaptation/{task_name}/MSE', mse_loss, k+1)
                    writer.add_scalar(f'Adaptation/{task_name}/MAE', mae, k+1)
                    writer.add_scalar(f'Adaptation/{task_name}/RMSE', rmse, k+1)
                    writer.add_scalar(f'Adaptation/{task_name}/R2', r2, k+1)

        mses = np.array(mses)
        maes = np.array(maes)
        rmses = np.array(rmses)
        r2s = np.array(r2s)

        return net, fast_weights, mses, maes, rmses, r2s
    
    def evaluate(self, test_dataloader):
        """
        Evaluate the model on the given dataset
        :param dataloader: DataLoader
        :return: loss, accuracy
        """
        net = deepcopy(self.net)
        net.eval()
        total_loss = 0.0
        total_mae = 0.0
        for x, y in test_dataloader:
            x = x.to(net.device)
            y = y.to(net.device)
            logits = net(x)
            loss = F.mse_loss(logits, y)
            mae = torch.abs(logits - y).mean()
            total_loss += loss.item()
            total_mae += mae

        total_loss /= len(test_dataloader)
        total_mae /= len(test_dataloader)

        return total_loss, total_mae

    def get_state_dict(self):
        return self.net.state_dict()

    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)
        

class MAML_L2F(nn.Module):
    """
    MAML Learner
    """
    def __init__(self, args, learner, rep=0, writer=None):
        """

        :param args:
        """
        super(MAML_L2F, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.grad_clip = args.grad_clip
        self.second_order = args.second_order
        self.output_dir = args.output
        os.makedirs(os.path.join(self.output_dir, "weights"), exist_ok=True)
        self.rep = rep
        self.writer = writer

        # Definir el orden de los grupos según los nombres de los parámetros.
        # Para una ResNet1D, se asume que existen "conv1", "layer1", "layer2", "layer3", "layer4" y "fc".
        self.groups_order = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]

        # Red atenuadora: genera un factor gamma para cada grupo.
        self.attenuator = Attenuator(num_groups=len(self.groups_order)).to(args.device)

        self.net = learner
        self.meta_optim = optim.Adam(list(self.net.parameters()) + list(self.attenuator.parameters()), lr=self.meta_lr)        

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter
    
    def fast_weights(self, loss, params, lr):
        grad = torch.autograd.grad(loss, params)
        if self.grad_clip is not None:
            grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
        params = list(map(lambda p: p[1] - lr * p[0], zip(grad, params)))
        return params
    
    def get_attenuated_params(self, net, x_spt, y_spt):
        # --- L2F: Calcular la inicialización atenuada ---
        # Usamos explícitamente la lista de parámetros registrados.
        base_params = list(net.parameters())
        base_named_params = dict(net.named_parameters())
        # Forward en el support set pasando los parámetros base.
        logits_support = net(x_spt, vars=base_params, bn_training=True)
        loss_support = F.mse_loss(logits_support, y_spt)

        # Calcular los gradientes respecto a base_params.
        grads = torch.autograd.grad(loss_support, base_params, create_graph=self.second_order)

        # Agrupar los parámetros por capa usando los nombres.
        grouped = group_params_by_layer(net, self.groups_order)
        task_embedding_list = []
        for group in self.groups_order:
            grad_means = []
            for name, param in grouped[group]:
                keys = list(base_named_params.keys())
                idx = keys.index(name)
                grad_means.append(grads[idx].mean())
            group_mean = torch.stack(grad_means).mean()
            task_embedding_list.append(group_mean)
        task_embedding = torch.stack(task_embedding_list)  # Vector de dimensión (num_groups,)
        # Obtener los factores de atenuación (gamma) para cada grupo.
        gamma = self.attenuator(task_embedding)  # Tensor de tamaño (num_groups,)

        # Aplicar atenuación a cada parámetro según su grupo.
        attenuated_params_dict = {}
        grouped = group_params_by_layer(net, self.groups_order)
        for group_idx, group in enumerate(self.groups_order):
            for name, param in grouped[group]:
                attenuated_params_dict[name] = param * gamma[group_idx]
        # Reconstruir la lista de parámetros atenuados en el mismo orden que en named_parameters().
        attenuated_params = []
        for name, param in net.named_parameters():
            if name in attenuated_params_dict:
                attenuated_params.append(attenuated_params_dict[name])
            else:
                if "vars_bn." not in name:
                    attenuated_params.append(param)

        return attenuated_params


    def forward(self, dataset, save_weights=False, task_subset=None):
        task_num = len(dataset)
        suppsz, querysz = dataset.supp_sz, dataset.query_sz

        if task_subset is None:
            iterate_over = range(task_num)
        else:
            iterate_over = task_subset

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        maes = [0 for _ in range(self.update_step + 1)]
        mses = [0 for _ in range(self.update_step + 1)]
        rmse = [0 for _ in range(self.update_step + 1)]
        r2s = [0 for _ in range(self.update_step + 1)]


        for i in iterate_over:
            task = dataset[i]
            sampled_data = task.sample(suppsz, querysz)
            x_spt = sampled_data['support_features']
            y_spt = sampled_data['support_targets']
            x_qry = sampled_data['query_features']
            y_qry = sampled_data['query_targets']

            # # --- L2F: Calcular la inicialización atenuada ---
            # # Usamos explícitamente la lista de parámetros registrados.
            # base_params = list(self.net.parameters())
            # base_named_params = dict(self.net.named_parameters())
            # # Forward en el support set pasando los parámetros base.
            # logits_support = self.net(x_spt, vars=base_params, bn_training=True)
            # loss_support = F.mse_loss(logits_support, y_spt)

            # # Calcular los gradientes respecto a base_params.
            # grads = torch.autograd.grad(loss_support, base_params, create_graph=self.second_order)

            # # Agrupar los parámetros por capa usando los nombres.
            # grouped = group_params_by_layer(self.net, self.groups_order)
            # task_embedding_list = []
            # for group in self.groups_order:
            #     grad_means = []
            #     for name, param in grouped[group]:
            #         keys = list(base_named_params.keys())
            #         idx = keys.index(name)
            #         grad_means.append(grads[idx].mean())
            #     group_mean = torch.stack(grad_means).mean()
            #     task_embedding_list.append(group_mean)
            # task_embedding = torch.stack(task_embedding_list)  # Vector de dimensión (num_groups,)
            # # Obtener los factores de atenuación (gamma) para cada grupo.
            # gamma = self.attenuator(task_embedding)  # Tensor de tamaño (num_groups,)

            # # Aplicar atenuación a cada parámetro según su grupo.
            # attenuated_params_dict = {}
            # grouped = group_params_by_layer(self.net, self.groups_order)
            # for group_idx, group in enumerate(self.groups_order):
            #     for name, param in grouped[group]:
            #         attenuated_params_dict[name] = param * gamma[group_idx]
            # # Reconstruir la lista de parámetros atenuados en el mismo orden que en named_parameters().
            # attenuated_params = []
            # for name, param in self.net.named_parameters():
            #     if name in attenuated_params_dict:
            #         attenuated_params.append(attenuated_params_dict[name])
            #     else:
            #         if "vars_bn." not in name:
            #             attenuated_params.append(param)

            attenuated_params = self.get_attenuated_params(self.net, x_spt, y_spt)

            # Calcular los gradientes de loss_support respecto a la inicialización atenuada.
            logits_support_attenuated = self.net(x_spt, vars=attenuated_params, bn_training=True)
            loss_support_attenuated = F.mse_loss(logits_support_attenuated, y_spt)
            grad_attenuated = torch.autograd.grad(loss_support_attenuated, attenuated_params, create_graph=self.second_order)

            # grad_attenuated = torch.autograd.grad(loss_support, attenuated_params, create_graph=self.second_order)
            fast_weights = [p - self.update_lr * g for p, g in zip(attenuated_params, grad_attenuated)]
            # --- Fin de L2F ---

            # # 1. run the i-th task and compute loss for k=0
            # logits = self.net(x_spt, vars=None, bn_training=True)
            # loss = F.mse_loss(logits, y_spt)
            # grad = torch.autograd.grad(loss, self.net.parameters(), retain_graph=self.second_order, create_graph=self.second_order)
            # if self.grad_clip is not None:
            #     grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
            # fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            # fast_weights = self.fast_weights(loss, self.net.parameters(), self.update_lr)

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry, list(self.net.parameters()), bn_training=True)
                loss_q = F.mse_loss(logits_q, y_qry)
                losses_q[0] += loss_q

                mae_loss = torch.abs(logits_q - y_qry).mean().detach().cpu()
                maes[0] = maes[0] + mae_loss
                mses[0] = mses[0] + loss_q.item()
                rmse[0] = rmse[0] + torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
                r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
                r2s[0] = r2s[0] + r2

            # this is the loss and accuracy after the first update
            # [setsz, nway]
            logits_q = self.net(x_qry, fast_weights, bn_training=True)
            loss_q = F.mse_loss(logits_q, y_qry)
            losses_q[1] += loss_q
            with torch.no_grad():                
                # [setsz]
                mae_loss = torch.abs(logits_q - y_qry).mean().detach().cpu()
                maes[1] = maes[1] + mae_loss
                mses[1] = mses[1] + loss_q.item()
                rmse[1] = rmse[1] + torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
                r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
                r2s[1] = r2s[1] + r2

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt, fast_weights, bn_training=True)
                loss = F.mse_loss(logits, y_spt)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=self.second_order, create_graph=self.second_order)
                if self.grad_clip is not None:
                    grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                # fast_weights = self.fast_weights(loss, fast_weights, self.update_lr)

                logits_q = self.net(x_qry, fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.mse_loss(logits_q, y_qry)
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    mae_loss = torch.abs(logits_q - y_qry).mean().detach().cpu()
                    maes[k + 1] = maes[k + 1] + mae_loss
                    mses[k + 1] = mses[k + 1] + loss_q.item()
                    rmse[k + 1] = rmse[k + 1] + torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
                    r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
                    r2s[k + 1] = r2s[k + 1] + r2

            # Save adapted weights
            if save_weights:
                pickle.dump(fast_weights, open(os.path.join(self.output_dir, "weights", f'adapted_weights_{self.rep}_{task.name}.pkl'), 'wb'))

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # from torchviz import make_dot
        # dot = make_dot(loss_q, params=dict(self.net.named_parameters()))
        # dot.format = 'png'
        # dot.render('computational_graph')

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        if self.grad_clip is not None:
            for p in self.net.parameters():
                if p.grad is not None:
                    p.grad = torch.clamp_(p.grad, -self.grad_clip, self.grad_clip)
                else:
                    p.grad = torch.zeros_like(p)
        self.meta_optim.step()

        # mses = np.array(mses) / (querysz * task_num)
        # maes = np.array(maes) / (querysz * task_num)

        mses = np.array(mses) / task_num
        maes = np.array(maes) / task_num
        rmse = np.array(rmse) / task_num
        r2s = np.array(r2s) / task_num

        return mses, maes, rmse, r2s


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 3

        querysz = x_qry.size(0)

        maes = [0 for _ in range(self.update_step_test + 1)]
        mses = [0 for _ in range(self.update_step_test + 1)]
        rmses = [0 for _ in range(self.update_step_test + 1)]
        r2s = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        attenuated_params = self.get_attenuated_params(net, x_spt, y_spt)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.mse_loss(logits, y_spt)
        grad = torch.autograd.grad(loss, attenuated_params, retain_graph=self.second_order, create_graph=self.second_order)
        if self.grad_clip is not None:
            grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, attenuated_params)))
        # fast_weights = self.fast_weights(loss, attenuated_params, self.update_lr)

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, list(attenuated_params), bn_training=True)
            # [setsz]
            mse_loss = F.mse_loss(logits_q, y_qry).item()
            mses[0] = mses[0] + mse_loss
            mae = torch.abs(logits_q - y_qry).mean().detach().cpu()
            maes[0] = maes[0] + mae
            rmse = torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
            rmses[0] = rmses[0] + rmse
            r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
            r2s[0] = r2s[0] + r2

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            mse_loss = F.mse_loss(logits_q, y_qry).item()
            mses[1] = mses[1] + mse_loss
            mae = torch.abs(logits_q - y_qry).mean().detach().cpu()
            maes[1] = maes[1] + mae
            rmse = torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
            rmses[1] = rmses[1] + rmse
            r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
            r2s[1] = r2s[1] + r2

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.mse_loss(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights, retain_graph=self.second_order, create_graph=self.second_order)
            if self.grad_clip is not None:
                grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            # fast_weights = self.fast_weights(loss, fast_weights, self.update_lr)

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.mse_loss(logits_q, y_qry)

            with torch.no_grad():
                mse_loss = loss_q.item()
                mses[k+1] = mses[k+1] + mse_loss
                mae = torch.abs(logits_q - y_qry).mean().detach().cpu()
                maes[k+1] = maes[k+1] + mae
                rmse = torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
                rmses[k+1] = rmses[k+1] + rmse
                r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
                r2s[k+1] = r2s[k+1] + r2


        # del net

        mses = np.array(mses)
        maes = np.array(maes)
        rmses = np.array(rmses)
        r2s = np.array(r2s)

        return net, fast_weights, mses, maes, rmses, r2s

    def finetuning_batched(self, x_spt, y_spt, x_qry, y_qry, batch_size, writer=None, task_name=None):
        """
        Finetuning with minibatches
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :param batch_size: size of each minibatch
        :return:
        """
        assert len(x_spt.shape) == 3

        querysz = x_qry.size(0)

        maes = [0 for _ in range(self.update_step_test + 1)]
        mses = [0 for _ in range(self.update_step_test + 1)]
        rmses = [0 for _ in range(self.update_step_test + 1)]
        r2s = [0 for _ in range(self.update_step_test + 1)]

        net = deepcopy(self.net)

        fast_weights = self.get_attenuated_params(net, x_spt, y_spt)

        def get_batches(x, y, batch_size):
            for i in range(0, len(x), batch_size):
                yield x[i:i + batch_size], y[i:i + batch_size]

        # 1. run the i-th task and compute loss for k=0
        fast_weights = list(net.parameters())
        for x_spt_batch, y_spt_batch in get_batches(x_spt, y_spt, batch_size):
            logits = net(x_spt_batch, fast_weights, bn_training=True)
            loss = F.mse_loss(logits, y_spt_batch)
            grad = torch.autograd.grad(loss, fast_weights, retain_graph=False, create_graph=False)
            if self.grad_clip is not None:
                grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, list(net.parameters()), bn_training=True)
            # [setsz]
            mse_loss = F.mse_loss(logits_q, y_qry).item()
            mses[0] = mses[0] + mse_loss
            mae = torch.abs(logits_q - y_qry).mean().detach().cpu()
            maes[0] = maes[0] + mae
            rmse = torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
            rmses[0] = rmses[0] + rmse
            r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
            r2s[0] = r2s[0] + r2
            if writer is not None:
                writer.add_scalar(f'Adaptation/{task_name}/MSE', mse_loss, 0)
                writer.add_scalar(f'Adaptation/{task_name}/MAE', mae, 0)
                writer.add_scalar(f'Adaptation/{task_name}/RMSE', rmse, 0)
                writer.add_scalar(f'Adaptation/{task_name}/R2', r2, 0)


        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            mse_loss = F.mse_loss(logits_q, y_qry).item()
            mses[1] = mses[1] + mse_loss
            mae = torch.abs(logits_q - y_qry).mean().detach().cpu()
            maes[1] = maes[1] + mae
            rmse = torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
            rmses[1] = rmses[1] + rmse
            r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
            r2s[1] = r2s[1] + r2
            if writer is not None:
                writer.add_scalar(f'Adaptation/{task_name}/MSE', mse_loss, 1)
                writer.add_scalar(f'Adaptation/{task_name}/MAE', mae, 1)
                writer.add_scalar(f'Adaptation/{task_name}/RMSE', rmse, 1)
                writer.add_scalar(f'Adaptation/{task_name}/R2', r2, 1)

        for k in range(1, self.update_step_test):
            for x_spt_batch, y_spt_batch in get_batches(x_spt, y_spt, batch_size):
                logits = net(x_spt_batch, fast_weights, bn_training=True)
                loss = F.mse_loss(logits, y_spt_batch)
                grad = torch.autograd.grad(loss, fast_weights, retain_graph=False, create_graph=False)
                if self.grad_clip is not None:
                    grad = [torch.clamp(g, -self.grad_clip, self.grad_clip) for g in grad]
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.mse_loss(logits_q, y_qry)

            with torch.no_grad():
                mse_loss = loss_q.item()
                mses[k+1] = mses[k+1] + mse_loss
                mae = torch.abs(logits_q - y_qry).mean().detach().cpu()
                maes[k+1] = maes[k+1] + mae
                rmse = torch.sqrt(F.mse_loss(logits_q, y_qry)).item()
                rmses[k+1] = rmses[k+1] + rmse
                r2 = 1 - (torch.sum((logits_q - y_qry) ** 2) / torch.sum((y_qry - y_qry.mean()) ** 2)).item()
                r2s[k+1] = r2s[k+1] + r2
                if writer is not None:
                    writer.add_scalar(f'Adaptation/{task_name}/MSE', mse_loss, k+1)
                    writer.add_scalar(f'Adaptation/{task_name}/MAE', mae, k+1)
                    writer.add_scalar(f'Adaptation/{task_name}/RMSE', rmse, k+1)
                    writer.add_scalar(f'Adaptation/{task_name}/R2', r2, k+1)

        mses = np.array(mses)
        maes = np.array(maes)
        rmses = np.array(rmses)
        r2s = np.array(r2s)

        return net, fast_weights, mses, maes, rmses, r2s

    def evaluate(self, test_dataloader):
        """
        Evaluate the model on the given dataset
        :param dataloader: DataLoader
        :return: loss, accuracy
        """
        net = deepcopy(self.net)
        net.eval()
        total_loss = 0.0
        total_mae = 0.0
        for x, y in test_dataloader:
            x = x.to(net.device)
            y = y.to(net.device)
            logits = net(x)
            loss = F.mse_loss(logits, y)
            mae = torch.abs(logits - y).mean()
            total_loss += loss.item()
            total_mae += mae

        total_loss /= len(test_dataloader)
        total_mae /= len(test_dataloader)

        return total_loss, total_mae

    def get_state_dict(self):
        return self.net.state_dict()

    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)


def main():
    pass


if __name__ == '__main__':
    main()