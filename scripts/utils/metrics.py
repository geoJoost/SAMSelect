from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, \
    cohen_kappa_score, jaccard_score, accuracy_score

# Code adapted from: https://github.com/MarcCoru/marinedebrisdetector/blob/44f7f5898f37b4b1bfb3378a0ef5e587a3e3ead2/marinedebrisdetector/metrics.py
def calculate_metrics(targets, predictions):
    # Pre-processing
    targets[targets == 255] = 1 # Convert [0, 255] => [0, 1]
    y_true = targets.view(-1)
    y_pred = predictions.view(-1)

    #predictions = scores > optimal_threshold

    #auroc = roc_auc_score(targets, scores)
    p, r, f, s = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, zero_division=0, average="binary")
    
    kappa = cohen_kappa_score(y_true, y_pred)

    jaccard = jaccard_score(y_true, y_pred)

    accuracy = accuracy_score(y_true, y_pred)

    summary = dict(
        #auroc=auroc,
        precision=p,
        accuracy=accuracy,
        recall=r,
        fscore=f,
        kappa=kappa,
        jaccard=jaccard,
        #threshold=optimal_threshold
    )

    return summary