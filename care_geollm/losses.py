import torch
import torch.nn.functional as F

def heteroscedastic_nll(pred, target, log_scale, weights=None):
    term=((target-pred)**2)/(2*torch.exp(2*log_scale)) + log_scale
    if weights is not None: term=term*weights
    return term.mean()

def untreated_consistency(h0,hp,untreated_mask=None):
    if untreated_mask is None or untreated_mask.sum()==0: return h0.new_tensor(0.)
    return ((hp[untreated_mask]-h0[untreated_mask])**2).mean()

def alignment_loss(masks,target=None):
    if target is None: return masks.new_tensor(0.)
    # tolerate targets with fewer/more intervention rows by aligning the common prefix
    k=min(masks.shape[0],target.shape[0]); n=min(masks.shape[1],target.shape[1])
    return F.binary_cross_entropy(masks[:k,:n].clamp(1e-5,1-1e-5),target[:k,:n])

def direction_loss(delta, direction_sign=None):
    if direction_sign is None:
        return F.relu(-delta).mean()
    sign=direction_sign.to(delta.device)
    known=sign!=0
    if not known.any(): return delta.new_tensor(0.)
    return F.relu(-(delta[known]*sign[known])).mean()

def total_loss(out,batch,cfg,target_weights=None,paraphrase_delta=None,direction_sign=None):
    if getattr(out.get('model_config',None),'use_uncertainty_head',True):
        pass
    use_uncertainty = getattr(cfg, 'use_uncertainty_head', None)
    # TrainConfig does not own the model switch; infer from whether scales require/use variation.
    if use_uncertainty is False:
        lp=F.mse_loss(out['delta'],batch.target_delta)
    else:
        lp=heteroscedastic_nll(out['delta'],batch.target_delta,out['scale'],target_weights)
    la=alignment_loss(out['masks'],batch.align_target)
    lc=untreated_consistency(out['h0'],out['hp'],batch.untreated_mask)
    ld=direction_loss(out['delta'],direction_sign)
    lpara=out['delta'].new_tensor(0.) if paraphrase_delta is None else F.mse_loss(out['delta'],paraphrase_delta.detach())
    total=lp+cfg.lambda_align*la+cfg.lambda_cons*lc+cfg.lambda_para*lpara+cfg.lambda_dir*ld
    return total, {'pred':float(lp.detach()),'align':float(la.detach()),'cons':float(lc.detach()),'para':float(lpara.detach()),'dir':float(ld.detach())}
