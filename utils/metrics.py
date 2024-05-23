from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import numpy as np

def metrics_cls(pred_li, target_li, demo_li):
    pred_epochi, target_epochi, demo_epochi = pred_li.cpu().detach().numpy(), target_li.cpu().detach().numpy(), demo_li.cpu().detach().numpy()
    y_pred = np.argmax(pred_epochi, axis=1)
    accuracy = accuracy_score(target_epochi, y_pred)
    precision = precision_score(target_epochi, y_pred)
    recall = recall_score(target_epochi, y_pred)
    f1 = f1_score(target_epochi, y_pred)
    y_proba =  pred_epochi - np.max(pred_epochi, axis= 1, keepdims=True)
    y_proba = np.exp(y_proba) / np.sum(np.exp(y_proba), axis=1, keepdims=True)
    auroc = roc_auc_score(target_epochi, y_proba[:,1])
    auprc = average_precision_score(target_epochi, y_proba[:,1])
    precision_macro = precision_score(target_epochi, y_pred, average='macro')
    recall_macro = recall_score(target_epochi, y_pred, average='macro')
    f1_macro = f1_score(target_epochi, y_pred, average='macro')
    metrics_res = {'acc': accuracy,
                   'precision': precision,
                   'recall': recall,
                   'f1': f1,
                   'auroc': auroc,
                   'auprc': auprc,
                   'precision_macro': precision_macro,
                   'recall_macro': recall_macro,
                   'f1_macro': f1_macro
                   }
    return metrics_res