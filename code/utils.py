from typing import List, Union, Optional

import torch


from typing import List, Union, Optional
import torch

def padded_tensor(
    items: List[Union[List[int], torch.LongTensor]],
    pad_idx: int = 0,
    pad_tail: bool = True,
    max_len: Optional[int] = None,
    use_amp: bool = False
) -> torch.LongTensor:
    """
    A device-agnostic tensor padding function that always creates tensors on CPU.
    """
    n = len(items)
    lens: List[int] = [len(item) for item in items]
    t = max(lens) if lens else 0
    t = max(t, 1)  # Ensure the tensor has at least length 1

    if max_len is not None:
        t = min(t, max_len)
    
    # The logic of use_amp usually does not need to be handled manually in FSDP and DataLoader, but we keep the original behavior here
    if use_amp and t > 8:
        t = (t // 8) * 8
    
    # Remove the device argument and create tensors on CPU by default
    output = torch.full((n, t), fill_value=pad_idx, dtype=torch.long)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        
        # Ensure item is a tensor on CPU
        if not isinstance(item, torch.Tensor):
            # Remove the device argument
            item = torch.tensor(item, dtype=torch.long)
        
        # Move item to CPU just in case
        item = item.to('cpu')

        if pad_tail:
            output[i, :min(length, t)] = item[:t]
        else:
            output[i, t - min(length, t):] = item[-t:]

    return output



def padded_tensor2(
    items: List[Union[List[int], torch.LongTensor]],
    pad_idx: int = 0,
    pad_tail: bool = True,
    max_len: Optional[int] = None,
    debug: bool = False,
    device: torch.device = torch.device('cpu'),
    use_amp: bool = False
) -> torch.LongTensor:
    """Create a padded matrix from an uneven list of lists.

    Returns padded matrix.

    Matrix is right-padded (filled to the right) by default, but can be
    left padded if the flag is set to True.

    Matrix can also be placed on cuda automatically.

    :param list[iter[int]] items: List of items
    :param int pad_idx: the value to use for padding
    :param bool pad_tail:
    :param int max_len: if None, the max length is the maximum item length

    :returns: padded tensor.
    :rtype: Tensor[int64]

    """
    # number of items
    n = len(items)
    # length of each item
    lens: List[int] = [len(item) for item in items]
    # max in time dimension
    t = max(lens)
    # if input tensors are empty, we should expand to nulls
    t = max(t, 1)
    if debug and max_len is not None:
        t = max(t, max_len)

    if use_amp:
        t = t // 8 * 8

    output = torch.full((n, t), fill_value=pad_idx, dtype=torch.long, device=device)

    for i, (item, length) in enumerate(zip(items, lens)):
        if length == 0:
            continue
        if not isinstance(item, torch.Tensor):
            item = torch.tensor(item, dtype=torch.long, device=device)
        if pad_tail:
            output[i, :length] = item
        else:
            output[i, t - length:] = item

    return output

