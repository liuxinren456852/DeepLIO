import torch


def deeplio_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    # IMU and ground-truth measurments can have different length btw. each pair of lidar frames,
    # so we do not change their size and let them as their are
    imus = [b['imus'] for b in batch]
    gts = torch.stack([b['gts'] for b in batch])

    # also vladiation field and meta-datas need not to be changed
    metas = [b['meta'] for b in batch]

    res ={}
    res['imus'] = imus
    res['gts'] = gts
    res['metas'] = metas
    return res

