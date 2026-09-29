import argparse, torch
from care_geollm.config import ModelConfig,TrainConfig
from care_geollm.model import CAREGeoLLM
from care_geollm.synthetic_data import make_sample
from care_geollm.losses import total_loss
from care_geollm.utils import seed_everything

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--out',default='checkpoint.pt'); args=ap.parse_args()
    mc=ModelConfig(); tc=TrainConfig(epochs=args.epochs); seed_everything(tc.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=CAREGeoLLM(mc).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=tc.spatial_lr,weight_decay=tc.weight_decay)
    for epoch in range(tc.epochs):
        batch=make_sample(mc,seed=tc.seed+epoch)
        for name,val in vars(batch).items():
            if torch.is_tensor(val): setattr(batch,name,val.to(device))
        optim.zero_grad(); out=model(batch); loss,parts=total_loss(out,batch,tc); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),tc.grad_clip); optim.step()
        print(f'epoch={epoch+1} loss={loss.item():.4f} delta={out["delta"].detach().cpu().tolist()} parts={parts}')
    torch.save({'model':model.state_dict(),'model_config':vars(mc)},args.out); print('saved',args.out)
if __name__=='__main__': main()
