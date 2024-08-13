import torch
from torchmetrics.functional import r2_score
import pandas as pd
import numpy as np

def compute_r2(x, y):
    return r2_score(x, y)

def r2_loss(x, y):
    return 1 - compute_r2(x, y)

def generate_submission(gen_preds, mu, sigma, csv, submission_name):
    idx = csv.iloc[:, 0].values.tolist()

    y_hat = gen_preds()
    preds = (
        torch.cat(y_hat, dim=0).detach().cpu() # type: ignore
        if type(y_hat) is torch.Tensor else
        torch.tensor(y_hat)
    )
    preds = ((preds*sigma) + mu).numpy()

    df = pd.DataFrame(preds, columns=['X4', 'X11', 'X18', 'X26', 'X50', 'X3112'])
    df.insert(0, 'id', idx)

    submission_path = "submissions/" + submission_name + ".csv"
    df.to_csv(submission_path, index=False)