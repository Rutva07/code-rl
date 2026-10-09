"""GRPO loss shared by edit actions and causal-LM output tokens."""
import torch


def group_advantages(rewards, eps=1e-6):
    if rewards.ndim != 1 or len(rewards) < 2:
        raise ValueError("each GRPO group requires at least two rewards")
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(eps)


def grpo_loss(logp, old_logp, ref_logp, rewards, mask=None, clip=0.2, beta=0.01):
    """Each row is one sampled completion; mask excludes prompt/padding tokens."""
    if logp.ndim == 1:
        logp, old_logp, ref_logp = [x[:, None] for x in (logp, old_logp, ref_logp)]
    mask = torch.ones_like(logp) if mask is None else mask.to(logp)
    advantages = group_advantages(rewards).detach()[:, None]
    ratio = (logp - old_logp.detach()).exp()
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1-clip, 1+clip) * advantages)
    delta = ref_logp.detach() - logp
    kl = delta.exp() - delta - 1
    per_completion = ((-surrogate + beta * kl) * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return per_completion.mean(), {"kl": float((kl.detach() * mask).sum() / mask.sum().clamp_min(1)),
                                   "reward_std": float(rewards.std(unbiased=False))}
