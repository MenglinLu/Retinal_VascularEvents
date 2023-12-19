import math
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import _LRScheduler

def Metrics_reg(predict_li, label_li):
    predict_li = predict_li.cpu().detach().numpy()
    label_li = label_li.cpu().detach().numpy()
    explained_variance_score1 = explained_variance_score(label_li, predict_li)
    mse_score = mean_squared_error(label_li, predict_li)
    rmse_score = math.sqrt(mse_score)
    mae_score = mean_absolute_error(label_li, predict_li)
    r2_score1 = r2_score(label_li, predict_li)
    return {'explained_variance_score': round(explained_variance_score1,4),
            'mse_score': round(mse_score,4),
            'rmse_score': round(rmse_score,4),
            'mae_score': round(mae_score,4),
            'r2_score': round(r2_score1,4)}

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
    
