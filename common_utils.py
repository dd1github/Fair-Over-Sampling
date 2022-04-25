# Metrics function
#Code adapted from https://github.com/Trusted-AI/AIF360



from collections import OrderedDict
from aif360.metrics import ClassificationMetric
import numpy as np

def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                            dataset_pred, 
                            unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                            classified_metric_pred.true_negative_rate())
    
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()
    
    metrics["Absolute average odds difference"] = classified_metric_pred.average_abs_odds_difference()
    metrics["True positive rate difference"] = classified_metric_pred.true_positive_rate_difference()
    metrics["True negative rate difference"] = classified_metric_pred.true_negative_rate_difference()
    
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
    
    metrics["Fair utility"] = metrics["Balanced accuracy"] * .5 * \
        ((1-np.abs(metrics["True positive rate difference"])) + \
         (1-np.abs(metrics["True negative rate difference"])))
    
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))
    
    return metrics