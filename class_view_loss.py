import torch
import torch.nn.functional as F


def class_loss(logits_mlp, logits_gnn, temperature):
    # Taking the soft label loss function as an example, compute softmax probabilities over the class-view dimension.
    pred_mlp = F.softmax(logits_mlp / temperature, dim=1)
    pred_gnn = F.softmax(logits_gnn / temperature, dim=1)

    loss = F.kl_div(
        pred_mlp,
        pred_gnn,
        reduction='sum'
    ) * (temperature ** 2)

    return loss