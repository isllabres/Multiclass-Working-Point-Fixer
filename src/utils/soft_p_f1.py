import torch


def soft_p_f1_loss(model_output: torch.Tensor,
                   target: torch.Tensor,
                   working_point: float) -> torch.Tensor:
    epsilon = 1e-7

    target_multiclass = torch.zeros((target.shape[0], 2))
    target_multiclass[:, 1] = target
    target_multiclass[:, 0] = torch.sub(1, target)

    working_point_p = torch.sub(torch.mul(working_point, 2), 1)
    proportion = torch.div(torch.add(1, working_point_p), torch.sub(1, working_point_p))

    vector_w = torch.ones(2)
    vector_w[1] = torch.mul(vector_w[1], torch.div(1, torch.add(proportion, epsilon)))

    model_output_multiclass = torch.ones((model_output.shape[0], 2))
    model_output_multiclass[:, 1] = model_output
    model_output_multiclass[:, 0] = torch.sub(1, model_output)

    model_output_multiclass_w = torch.mul(model_output_multiclass, vector_w)
    model_output_multiclass_w = torch.div(model_output_multiclass_w,
                                          model_output_multiclass_w.sum(axis=1).reshape(len(model_output_multiclass_w),
                                                                                        1))

    p_tp = (target_multiclass * model_output_multiclass_w).sum(axis=0).to(torch.float32)
    p_tn = ((1.0 - target_multiclass) * (1.0 - model_output_multiclass_w)).sum(axis=0).to(torch.float32)
    p_fp = ((1.0 - target_multiclass) * model_output_multiclass_w).sum(axis=0).to(torch.float32)
    p_fn = (target_multiclass * (1.0 - model_output_multiclass_w)).sum(axis=0).to(torch.float32)

    p_precision = p_tp / (p_tp + p_fp + epsilon)
    p_recall = p_tp / (p_tp + p_fn + epsilon)

    p_f1 = 2.0 * (p_precision * p_recall) / (p_precision + p_recall + epsilon)

    p_f1 = p_f1.mean()  # Mean to compute the F1 score "average='macro'"

    return 1 - p_f1