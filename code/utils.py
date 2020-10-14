import torch

# implementation of the Gradient Penalty
# to ensure that 1-Lipschitz continuity is met
def GradientPenaltyLoss(D, real_samples, fake_samples, reduction='mean', device=None):
        batch_size = len(real_samples)
        epsilon = torch.rand((batch_size, 1, 1, 1)).to(device)

        inputs = epsilon * real_samples + (1 - epsilon) * fake_samples
        inputs.requires_grad_(True)
        inputs = inputs.to(device)

        outputs = D(inputs)

        gradients = torch.autograd.grad(
            inputs=inputs,
            outputs=outputs,
            grad_outputs=torch.ones_like(outputs).to(device),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        
        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2

        reduction_func = None
        if reduction == 'mean':
            reduction_func = torch.mean
        elif reduction == 'sum':
            reduction_func = torch.sum
    
        return reduction_func(gradient_penalty)

# the implementation of the critic loss without the gradient penalty
def CriticLoss(fakes_preds, reals_preds, reduction='mean'):
    reduction_func = None
    if reduction == 'mean':
        reduction_func = torch.mean
    elif reduction == 'sum':
        reduction_func = torch.sum

    # the higher the score of fake predictions, the higher the loss -> because we want to predict as low as possible for fakes
    # the higher the score of real predictions, the lower the loss -> we want to predict as high as possible for fakes
    return reduction_func(fakes_preds) - reduction_func(reals_preds)

# the implementation of the generator loss
def GeneratorLoss(fakes_preds, reduction='mean'):
    reduction_func = None
    if reduction == 'mean':
        reduction_func = torch.mean
    elif reduction == 'sum':
        reduction_func = torch.sum
        
    # # we want to maximise the prediction of the critic on the fake samples
    return -1 * reduction_func(fakes_preds)