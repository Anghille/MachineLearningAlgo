import numpy as np
from ml.model.base import BaseModel

class r2_score():
    def __init__(self, pred, y, adjusted=False):
        self.y_pred = pred
        self.y_true = y
        self.adjusted = adjusted
    
    def compute(self):
        if not self.adjusted:
            sstot = np.sum((self.y_true - np.mean(self.y_true))**2)
            ssres = np.sum((self.y_true - self.y_pred)**2)
            return 1 - (ssres/sstot)
        else:
            n = 
            sstot = np.sum((self.y_true - np.mean(self.y_true))**2)
            ssres = np.sum((self.y_true - self.y_pred)**2)
            r_2 = 1 - (ssres/sstot)
            adj_r_2 = 1 - (1 - r_2)* ((n-1)/(n-p-1))
    
    