import torch
import numpy as np
import torch.nn.functional as F

from .utils.utils import empty_context

from sklearn.metrics import r2_score


class PrototypicalNetwork():
    def __init__(self, learner, device, meta_batch_size=1, dist_temperature=0.5, lr=0.001, momentum=0.9, opt_fn="adam"):
        super().__init__()
        self.dev = device
        self.meta_batch_size = meta_batch_size
        self.dist_temperature = dist_temperature
        self.lr = lr
        self.momentum = momentum
        self.opt_fn = opt_fn
        self.task_counter = 0
        self.baselearner = learner.to(self.dev)
        self.initialization = [
            p.clone().detach().to(self.dev) for p in self.baselearner.parameters()
        ]
        self.mode = "inverse_distance_weighted"
        for p in self.initialization:
            p.requires_grad = True

        if self.opt_fn == "sgd":
            self.optimizer = torch.optim.SGD(
                self.initialization, lr=self.lr, momentum=self.momentum
            )
        else:
            self.optimizer = torch.optim.Adam(self.initialization, lr=self.lr)

    def _calculate_distance(self, fts, prototype, scaler=20, task_type=None):
        """
        Calculate the distance between features and prototypes
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = torch.zeros(size=(fts.shape[0], prototype.shape[0])).cuda()
        for i, sample_query in enumerate(fts):
            dist[i] = (
                -torch.cdist(sample_query.unsqueeze(0), prototype)
                / self.dist_temperature
            )
        return dist

    def _get_features(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode="bilinear")
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) / (
            mask[None, ...].sum(dim=(2, 3)) + 1e-5
        )  # 1 x C
        return masked_fts

    def _get_prototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype
        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype

    def _deploy(
        self,
        train_x,
        train_y,
        test_x,
        test_y,
        train_mode,
        num_classes=None,
        task_type=None,
        embedding="after"
    ):
        if train_mode:
            contxt = empty_context
            num_classes = 1
        else:
            contxt = torch.no_grad
            if num_classes is None:
                num_classes = 1

        with contxt():
            if embedding == "after":
                support_embeddings = self.baselearner(
                    train_x, vars=self.initialization, embedding=True
                )
                query_embeddings = self.baselearner(
                    test_x, vars=self.initialization, embedding=True
                )
            elif embedding == "before":
                support_embeddings = self.baselearner(
                    train_x, vars=self.initialization, embedding_before=True
                )
                query_embeddings = self.baselearner(
                    test_x, vars=self.initialization, embedding_before=True
                )
        
            dists = self._calculate_distance(
                query_embeddings, support_embeddings, task_type="regression"
            )
            dists_norm = torch.nn.Softmax(dim=1)(dists)
            out = torch.mm(dists_norm, train_y)
            loss = F.mse_loss(out, test_y)
            
        with torch.no_grad():
            out_cpu = out.cpu().numpy()
            test_y_cpu = test_y.cpu().numpy()
            score = np.abs(out_cpu - test_y_cpu).mean()
            rmse = np.sqrt(np.mean((out_cpu - test_y_cpu)**2))
            r2 = r2_score(test_y_cpu, out_cpu)
            # score = regression_loss(out, test_y, task_type, mode="eval")
            preds = out.detach()
            probs = out.detach()

        return score, loss, rmse, r2, probs.cpu().numpy(), preds.cpu().numpy()

    def train(self, train_x, train_y, test_x, test_y, task_type, embedding="after"):
        self.baselearner.train()
        self.task_counter += 1

        train_x = train_x.to(self.dev)
        train_y = train_y.to(self.dev)
        test_x = test_x.to(self.dev)
        test_y = test_y.to(self.dev)

        score, loss, rmse, r2, probs, preds = self._deploy(
            train_x, train_y, test_x, test_y, True, task_type=task_type,
            embedding=embedding
        )

        loss.backward()
        if self.task_counter % self.meta_batch_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return score, loss.item(), rmse, r2, probs, preds


    def evaluate(
        self, num_classes, train_x, train_y, test_x, test_y, task_type, embedding="after", **kwargs
    ):
        self.baselearner.eval()
        train_x = train_x.to(self.dev)
        train_y = train_y.to(self.dev)
        test_x = test_x.to(self.dev)
        test_y = test_y.to(self.dev)

        score, loss, rmse, r2, probs, preds = self._deploy(
            train_x, train_y, test_x, test_y, False, num_classes, task_type=task_type,
            embedding=embedding
        )

        return score, loss.item(), rmse, r2, probs, preds

    def dump_state(self):
        return [p.clone().detach() for p in self.initialization]

    def load_state(self, state):
        self.initialization = [p.clone() for p in state]
        for p in self.initialization:
            p.requires_grad = True