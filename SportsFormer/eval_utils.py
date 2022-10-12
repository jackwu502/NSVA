import numpy as np

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def ref_success_rate(all_ref, pred, gt, aggregate=True):
    """required format
    Action space is a single integer
    train_test: The collection of [start, ..., goal] tuples from train & test dataset, format [num_of_instancs, pred_horz] in numpy;
    pred: The predicted intermediate action sequence, numpy [batch, seq];
    gt  : The ground-truth action label sequence    , numpy [batch, seq];

    Metric Procedure:
    "All" prediction steps has to match with gt steps
    """

    rst = []
    for i, j in zip(pred, gt):
        indices = np.where((all_ref[:, [0, -1]]==j[[0, -1]]).all(1))
        ref_set = np.unique(all_ref[indices], axis=0)

        if any((i == ref_set[:, :-1]).all(1)):
            rst.append(1.0)
        else:
            rst.append(0.0)

    return sum(rst)/ len(rst)


def success_rate(pred, gt, aggregate=True):
    """required format
    Action space is a single integer
    pred: The predicted intermediate action sequence, numpy [batch, seq];
    gt  : The ground-truth action label sequence    , numpy [batch, seq];

    Metric Procedure:
    "All" prediction steps has to match with gt steps
    """

    rst = np.all(np.equal(pred, gt), axis=(1)) 

    if aggregate:
        return np.mean(rst)
    else:
        return rst

def mean_category_acc(pred, gt):
    """required format
    Action space is a single integer
    pred: List [batch * seq]
    gt  : List [batch * seq]
    """

    from sklearn.metrics import precision_score
    rst = precision_score(gt, pred, average="macro")
    return rst

def acc_iou_onehot(pred, gt, aggregate=True):
    """required format
    Action space is a single integer
    pred: Numpy [batch, seq]
    gt  : Numpy [batch, seq]
    """
    epsn = 1e-6
    
    pred_onehot_list = []
    gt_onehot_list = []
    for p, g in zip(pred,gt):
        pred_onehot_list.append(one_hot(p, num_classes=40000))
        gt_onehot_list.append(one_hot(gt, num_classes=40000))
    pred = np.logical_or.reduce(np.stack(pred_onehot_list, 0))
    gt = np.logical_or.reduce(np.stack(gt_onehot_list, 0))
    
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
        
    return (intersection + epsn) / (union + epsn)

def acc_iou(pred, gt, aggregate=True):
    """required format
    Action space is a single integer
    pred: Numpy [batch, seq]
    gt  : Numpy [batch, seq]
    """

    epsn = 1e-6

    if aggregate:
        intersection = (pred & gt).sum((0, 1))
        union = (pred | gt).sum((0, 1))
    else:
        intersection = (pred & gt).sum((1))
        union = (pred | gt).sum((1))
        
    return (intersection + epsn) / (union + epsn)

if __name__ == "__main__":
    test_pred = [[0, 1, 2], [0, 2, 3]]
    gt_pred = [[0, 2, 3], [0, 1, 4]]

    print(mean_category_acc(test_pred[0], gt_pred[0]))
