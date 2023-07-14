import torch
from torch.distributions.categorical import Categorical
from torch.nn.functional import softmax
import numpy as np

def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torh.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill(
                (1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.

        # since BPP requires strict mask, set to 1e46 ensure it goes to zero prob
        vector = vector + (mask + 1e-46).log() 
    return torch.nn.functional.log_softmax(vector, dim=dim)


def obs2tensor(obs):
    return torch.FloatTensor(obs['state']), torch.FloatTensor(obs['mask'])

# def best_fit(items, capacity):

#     bins = np.zeros(len(items))
#     for item in items:
#         bin_res = capacity - bins - item
#         mask = bin_res < 0
#         idx = ma.masked_array(bin_res, mask).argmin(fill_value=100)
#         bins[idx] += item
#     bins_usage = np.sum((bins > 0).astype(np.int))
#     bin_levels = np.zeros(capacity).astype(int)
#     for b in bins:
#         if b > 0:
#             bin_levels[int(b)-1] += 1
#     total_waste = np.sum(capacity - bins) - (len(items) - bins_usage) * capacity
#     return bins_usage, total_waste, bin_levels





if __name__ == '__main__':

    vec = torch.FloatTensor([i for i in range(10, 20)])
    mask = torch.BoolTensor([1, 1, 0, 0, 0, 1, 1, 1, 1, 0])
    print(torch.nn.functional.softmax(masked_log_softmax(vec, mask)))
    print(sum(torch.nn.functional.softmax(masked_log_softmax(vec, mask))))
    logits = masked_log_softmax(vec, mask)
    # logits[logits < - 100] = -110
    print(logits)
    dist = Categorical(softmax(logits))
    # print(dist.probs)
    print(dist.probs)
    # print([dist.sample() for i in range(100)])
