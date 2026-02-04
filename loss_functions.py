import torch, math
import torch.nn as nn
import torch.nn.functional as F

class BCELossFn(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, out_model, batchs):
        logits = out_model["disease_logits"]

        if len(targets.shape) == 2:
            targets = torch.argmax(targets, dim=1).float()

        if len(logits.shape) == 2:
            logits = logits.squeeze(-1)

        losses = self.bce(logits, targets)
        return [losses]

class BCEPatContrasiveLossFn(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.temperature = temperature

    def forward(self, out_model, batchs):
        logits = out_model["disease_logits"]
        embeddings = out_model["embeddings"]
        _, _, _, _, targets, [patient_ids, _, _, _] = batchs 

        if len(targets.shape) == 2:
            targets = torch.argmax(targets, dim=1).float()

        if len(logits.shape) == 2:
            logits = logits.squeeze(-1)
            
        device = embeddings.device
        N = embeddings.size(0)

        # cosine similarity
        embeddings = F.normalize(embeddings, dim=1)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  # (N, N)

        # mask out self-similarity
        self_mask = torch.eye(N, device=device, dtype=torch.bool)
        sim.masked_fill_(self_mask, -1e9)

        # positives: same patient, different wav
        pid = patient_ids.unsqueeze(0)
        pos_mask = (pid == pid.T) & (~self_mask)

        # negatives: different patient
        neg_mask = pid != pid.T

        # log-softmax over all pairs
        log_prob = F.log_softmax(sim, dim=1)

        # only keep anchors that have positives
        valid = pos_mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        # InfoNCE
        loss = -(log_prob[valid] * pos_mask[valid]).sum(dim=1) / pos_mask[valid].sum(dim=1)
        InfoNCELoss = loss.mean()

        losses = self.bce(logits, targets)
        return [losses + 0.15 * InfoNCELoss]

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5, reduction="mean"):
        """
        alpha: class balancing factor (float or None)
        • Class ratio ≈ 1:1 → alpha ≈ 0.5
        • Positives ≈ 20–30% → alpha ≈ 0.3–0.4
        • Positives ≈ 5–10% → alpha ≈ 0.6–0.75
        • Extreme imbalance (<2%) → alpha ≈ 0.8–0.9

        gamma: focusing parameter
        • gamma = 0 → standard BCE
        • gamma ∈ [1,2] → mild hard-sample emphasis
        • gamma ≥ 3 → aggressive focus on misclassified / noisy samples

        • Clean labels, strong backbone → gamma = 2.0
        • Noisy labels (your cough/TB case) → gamma = 1.0–1.5
        • MIL / weak supervision → gamma = 0.5–1.5

        reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, out_model, batchs):
        """
        logits: (B,) or (B, 1) raw model outputs
        targets: (B,) binary labels {0,1}
        """
        logits = out_model["disease_logits"]
        _, _, _, _, targets, [patient_ids, _, _, _] = batchs 

        if len(targets.shape) == 2:
            targets = torch.argmax(targets, dim=1).float()

        if len(logits.shape) == 2:
            logits = logits.squeeze(-1)

        targets = targets.float()
        logits = logits.view(-1)
        targets = targets.view(-1)

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        p_t = torch.exp(-bce)  # = sigmoid(logits) if y=1 else 1-sigmoid
        focal_loss = (1 - p_t) ** self.gamma * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return [focal_loss.mean()]
        elif self.reduction == "sum":
            return [focal_loss.sum()]
        else:
            return [focal_loss]

class NormalizedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, out_model, batchs):
        logits = out_model["disease_logits"]
        _, _, _, _, targets, [patient_ids, _, _, _] = batchs 

        if len(targets.shape) == 2:
            targets = torch.argmax(targets, dim=1).float()

        if len(logits.shape) == 2:
            logits = logits.squeeze(-1)

        targets = targets.float().view(-1)
        logits = logits.view(-1)

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        p_t = torch.exp(-bce)
        focal = (1 - p_t) ** self.gamma * bce

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * focal

        # Normalization (key difference)
        return [focal.mean() / (focal.detach().mean() + self.eps)]


class ReverseCrossEntropy(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, out_model, batchs):
        """
        logits  : (B,) or (B, 1) raw outputs
        targets : (B,) or (B, 1) binary {0,1}
        """
        logits = out_model["disease_logits"]
        _, _, _, _, targets, [patient_ids, _, _, _] = batchs 

        if len(targets.shape) == 2:
            targets = torch.argmax(targets, dim=1).float()

        if len(logits.shape) == 2:
            logits = logits.squeeze(-1)

        logits = logits.view(-1)
        targets = targets.float().view(-1)

        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        # BCE-style Reverse Cross Entropy
        rce = -(
            probs * torch.log(targets + self.eps) +
            (1.0 - probs) * torch.log(1.0 - targets + self.eps)
        )
        return [rce.mean()]

class NFL_RCE_Loss(nn.Module):
    def __init__(self, lambda_rce=0.1):
        super().__init__()
        self.nfl = NormalizedFocalLoss()
        self.rce = ReverseCrossEntropy()
        self.lambda_rce = lambda_rce

    def forward(self, out_model, batchs):
        loss_nfl = self.nfl(out_model, batchs)[0]
        loss_rce = self.rce(out_model, batchs)[0]
        return [loss_nfl + self.lambda_rce * loss_rce]

# https://blog.allegro.tech/2023/04/learning-from-noisy-data.html#fn:2
class SPLBCELoss(nn.Module):
    def __init__(self, keep_ratio=0.7):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, batchs):
        if len(targets.shape) == 2:
            targets = torch.argmax(targets, dim=1)
            targets = (targets != 0).float()

        if len(logits.shape) == 2:
            logits = logits.squeeze(-1)

        losses = self.bce(logits, targets)  # (B,)

        B = losses.numel()
        k = max(1, int(self.keep_ratio * B))

        # select k smallest losses
        _, idx = torch.topk(losses, k, largest=False)
        selected_losses = losses[idx]

        return [selected_losses.mean()]

class PRLBCELoss(nn.Module):
    def __init__(self, keep_ratio=0.8):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, batchs):
        """
        logits: (B,)
        targets: (B,)
        """
        if len(targets.shape) == 2:
            targets = torch.argmax(targets, dim=1)
            targets = (targets != 0).float()

        if len(logits.shape) == 2:
            logits = logits.squeeze(-1)

        # per-sample BCE
        losses = self.bce(logits, targets)

        # gradient of BCE wrt logits (closed-form)
        probs = torch.sigmoid(logits)
        grad_norms = torch.abs(probs - targets)  # |∂ℓ/∂z|

        B = logits.numel()
        k = max(1, int(self.keep_ratio * B))

        # select samples with smallest gradient norms
        _, idx = torch.topk(grad_norms, k, largest=False)

        return [losses[idx].mean()]

class ClippedBCELoss(nn.Module):
    def __init__(self, clip_value=1.0):
        super().__init__()
        self.clip_value = clip_value
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, batchs):
        if len(targets.shape) == 2:
            targets = torch.argmax(targets, dim=1)
            targets = (targets != 0).float()

        if len(logits.shape) == 2:
            logits = logits.squeeze(-1)

        losses = self.bce(logits, targets)
        clipped_losses = torch.clamp(losses, max=self.clip_value)
        return [clipped_losses.mean()]

class ELRBCELoss(nn.Module):
    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        lambda_elr=3.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_elr = lambda_elr

        self.targets_ema = None
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def init_memory(self, num_samples):
        self.targets_ema = torch.zeros(num_samples)
        self.targets_ema = self.targets_ema.cuda()

    def forward(self, logits, targets, batchs):
        """
        logits: (B,)
        targets: (B,)
        indices: dataset indices of samples
        """
        if len(targets.shape) == 2:
            targets = torch.argmax(targets, dim=1)
            targets = (targets != 0).float()

        if len(logits.shape) == 2:
            logits = logits.squeeze(-1)

        _, _, _, _, _, [_, _, _, indices] = batchs 

        probs = torch.sigmoid(logits)

        # update EMA targets (no grad)
        with torch.no_grad():
            self.targets_ema[indices] = (
                self.beta * self.targets_ema[indices]
                + (1 - self.beta) * probs.detach()
            )

        # mixed targets
        mixed_targets = (
            (1 - self.alpha) * targets
            + self.alpha * self.targets_ema[indices]
        )

        # standard BCE
        bce_loss = self.bce(logits, mixed_targets)

        # ELR regularizer
        elr_reg = -torch.log(
            1 - probs * self.targets_ema[indices] + 1e-6
        )

        return [(bce_loss + self.lambda_elr * elr_reg).mean()]


class PatientAwareLoss(nn.Module):
    def __init__(self, agg="logsumexp"):
        super().__init__()
        assert agg in ["mean", "max", "logsumexp"]
        self.agg = agg
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, labels, batchs):
        """
        logits: (N_wavs, 1) or (N_wavs,)
        labels: (N_wavs,)  # same label for same patient
        patient_ids: (N_wavs,)
        """
        if len(labels.shape) == 2:
            labels = torch.argmax(labels, dim=1)
            labels = (labels != 0).float()

        if len(logits.shape) == 2:
            logits = logits.squeeze(-1)

        _, _, _, _, _, [patient_ids, _, _, indices] = batchs 
        logits = logits.squeeze()

        unique_pids = torch.unique(patient_ids)

        patient_logits = []
        patient_labels = []

        for pid in unique_pids:
            mask = patient_ids == pid

            bag_logits = logits[mask]
            bag_label = labels[mask][0]  # patient label

            if self.agg == "mean":
                z = bag_logits.mean()
            elif self.agg == "max":
                z = bag_logits.max()
            else:
                z = torch.logsumexp(bag_logits, dim=0)

            patient_logits.append(z)
            patient_labels.append(bag_label)

        patient_logits = torch.stack(patient_logits)
        patient_labels = torch.stack(patient_labels).float()
        return [self.bce(patient_logits, patient_labels)]

def get_losses_fn(loss_type="BCELossFn"):
    loss_cls = globals().get(loss_type)
    if loss_cls is None:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return loss_cls()



def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    # Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1

    # Check if labels[i] == labels[j]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

    mask = indices_not_equal * labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    # Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    # Check if labels[i] != labels[k]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check that i, j and k are distinct
    indices_not_same = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j * (i_equal_k ^ 1)

    mask = distinct_indices * valid_labels   # Combine the two masks

    return mask

class KLDivLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(KLDivLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y, t):
        # entropy = - torch.sum(t[t.data.nonzero()] * torch.log(t[t.data.nonzero()]))
        # crossEntropy = - torch.sum(t * F.log_softmax(y))
        # return (crossEntropy - entropy) / y.shape[0]
    
        # Ensure t is normalized
        t = t / t.sum(dim=1, keepdim=True).clamp(min=1e-12)

        # Apply log_softmax to predictions
        log_probs = F.log_softmax(y, dim=1)

        # KL divergence (manual version)
        kl = torch.sum(t * (torch.log(t.clamp(min=1e-12)) - log_probs), dim=1)

        if self.reduction == 'batchmean':
            return kl.mean()
        elif self.reduction == 'sum':
            return kl.sum()
        elif self.reduction == 'none':
            return kl
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ce = torch.nn.CrossEntropyLoss(weight=weight)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = self.alpha * (1 - p) ** self.gamma * logp
        return loss



class AngularMargin(nn.Module):
    """
    An implementation of Angular Margin (AM) proposed in the following
    paper: '''Margin Matters: Towards More Discriminative Deep Neural Network
    Embeddings for Speaker Recognition''' (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similarity
    scale : float
        The scale for cosine similarity

    Example
    -------
    >>> pred = AngularMargin()
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0):
        super().__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, outputs, targets):
        """Compute AM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Returns
        -------
        predictions : torch.Tensor
        """
        outputs = outputs - self.margin * targets
        return self.scale * outputs

class AdditiveAngularMargin(AngularMargin):
    """
    An implementation of Additive Angular Margin (AAM) proposed
    in the following paper: '''Margin Matters: Towards More Discriminative Deep
    Neural Network Embeddings for Speaker Recognition'''
    (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similarity.
    scale : float
        The scale for cosine similarity.
    easy_margin : bool

    Example
    -------
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> pred = AdditiveAngularMargin()
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        super().__init__(margin, scale)
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, outputs, targets):
        """
        Compute AAM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Returns
        -------
        predictions : torch.Tensor
        """
        cosine = outputs.float()
        cosine = torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs


class LogSoftmaxWrapper(nn.Module):
    """
    Arguments
    ---------
    loss_fn : Callable
        The LogSoftmax function to wrap.

    Example
    -------
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> outputs = outputs.unsqueeze(1)
    >>> targets = torch.tensor([ [0], [1], [0], [1] ])
    >>> log_prob = LogSoftmaxWrapper(nn.Identity())
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> log_prob = LogSoftmaxWrapper(AngularMargin(margin=0.2, scale=32))
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> log_prob = LogSoftmaxWrapper(AdditiveAngularMargin(margin=0.3, scale=32))
    >>> loss = log_prob(outputs, targets)
    >>> 0 <= loss < 1
    tensor(True)
    """

    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
        self.criterion = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets, length=None):
        """
        Arguments
        ---------
        outputs : torch.Tensor
            Network output tensor, of shape
            [batch, 1, outdim].
        targets : torch.Tensor
            Target tensor, of shape [batch, 1].
        length : torch.Tensor
            The lengths of the corresponding inputs.

        Returns
        -------
        loss: torch.Tensor
            Loss for current examples.
        """
        outputs = outputs.squeeze(1)
        targets = targets.squeeze(1)
        targets = F.one_hot(targets.long(), outputs.shape[1]).float()
        try:
            predictions = self.loss_fn(outputs, targets)
        except TypeError:
            predictions = self.loss_fn(outputs)

        predictions = F.log_softmax(predictions, dim=1)
        loss = self.criterion(predictions, targets) / targets.sum()
        return loss

class AFLoss(nn.Module):
    def __init__(self, margin, scale, num_layers):
        super().__init__()
        self.AAMLoss = LogSoftmaxWrapper(AdditiveAngularMargin(margin = margin, scale=scale))

    def forward(self, inputs, gate_weights, targets):
        targets = targets.unsqueeze(1)
        inputs = inputs.unsqueeze(1)
        labels = targets
        #gate_weights = gate_weights.sum(dim=0)
        loss = self.AAMLoss(inputs, labels) #+ (gate_weights.std()/gate_weights.mean())**2
        return loss

class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        labels = labels.flatten()
        weight = nn.functional.normalize(self.weight, dim=0)
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        cos_theta = torch.mm(hidden_states, weight)
        psi = cos_theta - self.margin

        onehot = nn.functional.one_hot(labels, self.num_labels)
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        loss = self.loss(logits, labels)

        return loss
    
class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss from Khosla et al. (2020):
    L = Σ_i 1/(|P(i)|) Σ_{p ∈ P(i)} -log( exp(z_i · z_p / τ) / Σ_{a≠i} exp(z_i · z_a / τ) )
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [B, D] embedding
        labels: [B] int labels
        """
        if labels.dim() == 2:
            labels = torch.argmax(labels, dim=1)
        device = features.device
        features = F.normalize(features, dim=1)

        logits = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # Mask out self-comparisons
        logits_mask = torch.ones_like(logits) - torch.eye(logits.size(0), device=device)
        logits = logits * logits_mask

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask * logits_mask  # remove diagonal

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        loss = -mean_log_prob_pos.mean()
        return loss

class CenterLoss(nn.Module):
    """
    Center Loss (Wen et al. 2016)
    """
    def __init__(self, num_classes=2, feat_dim=1024, lambda_c=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        
        # class centers: [C, D]
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        if labels.dim() == 2:
            labels = torch.argmax(labels, dim=1)
            
        batch_centers = self.centers[labels]  # [B, D]
        loss = ((features - batch_centers) ** 2).sum() / 2.0
        return self.lambda_c * loss

class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)

    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """
    def __init__(self, margin=0.1, hardest=False, squared=False):
        """
        Args:
            margin: margin for triplet loss
            hardest: If true, loss is considered only hardest triplets.
            squared: If true, output is the pairwise squared euclidean distance matrix.
                If false, output is the pairwise euclidean distance matrix.
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels):
        """
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)

        if self.hardest:
            # Get the hardest positive pairs
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

            # Get the hardest negative pairs
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

            # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + 0.1)
            triplet_loss = torch.mean(triplet_loss)
        else:
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)

            # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
            # triplet_loss[i, j, k] will contain the triplet loss of anc=i, pos=j, neg=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
            # and the 2nd (batch_size, 1, batch_size)
            loss = anc_pos_dist - anc_neg_dist + self.margin

            mask = _get_triplet_mask(labels).float()
            triplet_loss = loss * mask

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = F.relu(triplet_loss)

            # Count number of hard triplets (where triplet_loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)

            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss