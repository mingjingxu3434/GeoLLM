import argparse, torch
from care_geollm.config import ModelConfig
from care_geollm.model import CAREGeoLLM
from care_geollm.synthetic_data import make_sample

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default='checkpoint.pt'); args=ap.parse_args()
    mc=ModelConfig(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=CAREGeoLLM(mc).to(device)
    ck=torch.load(args.checkpoint,map_location=device); model.load_state_dict(ck['model']); model.eval()
    batch=make_sample(mc,seed=999)
    for n,v in vars(batch).items():
        if torch.is_tensor(v): setattr(batch,n,v.to(device))
    with torch.no_grad(): out=model(batch)
    names=['accessibility_gain','mobility_burden_reduction','operational_carbon_reduction','carbon_payback_improvement']
    for n,v,s in zip(names,out['delta'].cpu(),out['scale'].cpu()): print(f'{n}: effect={v.item():.4f}, log_scale={s.item():.4f}')
if __name__=='__main__': main()
