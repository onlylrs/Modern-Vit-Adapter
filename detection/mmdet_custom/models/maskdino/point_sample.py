import torch
from mmcv.ops import point_sample


def calculate_uncertainty(logits):
    assert logits.shape[1] == 1
    return -torch.abs(logits)


def get_uncertain_point_coords_with_randomness(mask_pred, num_points,
                                               oversample_ratio,
                                               importance_sample_ratio):
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    batch_size = mask_pred.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(batch_size, num_sampled, 2, device=mask_pred.device)
    point_logits = point_sample(mask_pred, point_coords, align_corners=False)
    point_uncertainties = calculate_uncertainty(point_logits)

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points,
                    dim=1)[1]
    shift = num_sampled * torch.arange(
        batch_size, dtype=torch.long, device=mask_pred.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        batch_size, num_uncertain_points, 2)
    if num_random_points > 0:
        rand_coords = torch.rand(batch_size, num_random_points, 2,
                                 device=mask_pred.device)
        point_coords = torch.cat((point_coords, rand_coords), dim=1)
    return point_coords
