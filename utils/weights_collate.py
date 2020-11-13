import torch
import os
import numpy as np
import random
import re
from torch._six import container_abcs, string_classes, int_classes
np_str_obj_array_pattern = re.compile(r'[SaUO]')

    # baseline distribution for one
def label2distribute(class_numbers,label, w_min: float = 1., w_max: float = 2e5):
    weight = np.ones(class_numbers, dtype='float32')
    N = np.prod(label.shape)
    for i in range(class_numbers):
        weight[i] =(np.sum(label == i)) / N #Make sure here should be same denominator, Otherwise the value is not allowed to be used to get the weight
    return weight

def label2_baseline_weight_by_prior(class_numbers,summary_factor,label, w_min: float = 1., w_max: float = 2e5):
    weight = np.zeros_like(label, dtype='float32')
    K = class_numbers - 1
    for i in range(class_numbers):
        weight[label == i] = 1 / (K + 1) * (1/(summary_factor[i]+2e-5))#modify to no zero divide
    # we add clip for learning stability
    # e.g. if we catch only 1 voxel of some component, the corresponding weight will be extremely high (~1e6)
    return np.clip(weight, w_min, w_max)

def distribution2tensor(class_numbers,summary_factor,label):
    weight = np.zeros_like(label, dtype='float32')
    K = class_numbers - 1
    for i in range(class_numbers):
        weight[label == i] = summary_factor[i]#modify to no zero divide
    # we add clip for learning stability
    # e.g. if we catch only 1 voxel of some component, the corresponding weight will be extremely high (~1e6)
    return weight

def default_collate_with_weight(batch):
    r"""
    Puts each data field into a tensor with outer dimension batch size
    (this is a modify version for  batch weight.)
    if the batch contain the key batch_xxx_weight, in this function we will statistic the weight for batch
    else call default_collate
    """
    elem = batch[0]
    # elem_type = type(elem)
    if isinstance(elem, container_abcs.Mapping):
        if 'batch_baseline_weight' in elem:# for baseline batch_weight
            num_class = elem['class_nums']
            avg_factor = np.zeros(num_class,)
            for item in batch:
                avg_factor += item['batch_baseline_weight']
            avg_factor /= len(batch)
            new_batch = []
            for item in batch:
                weight = label2_baseline_weight_by_prior(num_class,item['batch_baseline_weight'],item['mask'])
                new_batch.append({
                     'image': torch.from_numpy(item['image']).type(torch.FloatTensor),
                    'mask': torch.from_numpy(item['mask']).type(torch.IntTensor),
                    'weight':torch.from_numpy(weight).type(torch.FloatTensor)
                })
            re_batch = default_collate(new_batch)
            return re_batch
        elif 'batch_test_weight' in elem:# for collate batch_weight
            num_class = elem['class_nums']
            avg_factor = np.zeros(num_class,)
            for item in batch:
                avg_factor += item['batch_test_weight']
            avg_factor /= len(batch)
            new_batch = []
            for item in batch:
                label=item['mask']
                weight=np.zeros_like(label).astype(np.float)
                max_di = np.max(avg_factor)
                e = 2.7182
                for i in range(num_class):
                    pt = avg_factor[i]/max_di
                    weight[label == i] = 1.5*(1/e**pt)
                new_batch.append({
                     'image': torch.from_numpy(item['image']).type(torch.FloatTensor),
                    'mask': torch.from_numpy(item['mask']).type(torch.IntTensor),
                    'weight':torch.from_numpy(weight).type(torch.FloatTensor)
                })
            re_batch = default_collate(new_batch)
        elif 'batch_distrubution' in elem:# for collate batch_distribution
            num_class = elem['class_nums']
            avg_factor = np.zeros(num_class,)
            for item in batch:
               avg_factor += item['batch_test_weight']
            avg_factor /= len(batch)
            for item in batch:
                new_batch.append({
                     'image': torch.from_numpy(item['image']).type(torch.FloatTensor),
                    'mask': torch.from_numpy(item['mask']).type(torch.IntTensor),
                    'weight':torch.from_numpy(avg_factor.copy()).type(torch.FloatTensor)
                })
                re_batch = default_collate(new_batch)
            return re_batch
        else:
            return default_collate(batch)
    else:
        return default_collate(batch)

def default_collate(batch):
    """
    same as pytorch implement
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError('Unkown collate type in numpy object!' )

            return default_collate_with_weight([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate_with_weight([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate_with_weight(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate_with_weight(samples) for samples in transposed]

    raise TypeError('Unkown collate type')
