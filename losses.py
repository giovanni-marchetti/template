import torch 



def LpLoss(output_y, true_y, p=2):
    value = ((output_y - true_y) ** p).sum(-1) 
    return value.mean(0)


def AccuracyLoss(output_y, true_labels):
    out_labels = torch.argmax(output_y, dim=-1)
    value = (out_labels == true_labels)
    return value.mean(0) 

