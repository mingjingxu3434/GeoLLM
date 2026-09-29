import torch

class ConformalCalibrator:
    def __init__(self, coverage=0.90): self.coverage=coverage; self.q=None
    def fit(self,preds,targets,log_scales):
        scores=(targets-preds).abs()/torch.exp(log_scales).clamp_min(1e-6)
        self.q=torch.quantile(scores, self.coverage, dim=0)
        return self
    def interval(self,pred,log_scale):
        if self.q is None: raise RuntimeError('fit calibrator first')
        half=self.q.to(pred.device)*torch.exp(log_scale)
        return pred-half,pred+half
