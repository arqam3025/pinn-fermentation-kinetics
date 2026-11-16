import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).parent))  # one-case lambda sweep, see results/lambda_sensitivity.md
import pinn, numpy as np, torch
from concurrent.futures import ProcessPoolExecutor
true=dict(mu_max=0.15,Ks=0.6,Y=0.55)
rng=np.random.default_rng(0); t=np.sort(rng.uniform(1,16,10)); y=pinn.simulate(**true,t=t)*(1+0.05*rng.normal(size=(10,2)))
tg=np.linspace(0,40,200); yg=pinn.simulate(**true,t=tg); f=tg>16
def run(lam):
    torch.set_num_threads(1)
    net,est=pinn.train_nn(t,y,True,0,lam=lam)
    with torch.no_grad(): X=net(torch.tensor(tg,dtype=torch.float32)[:,None])[0].numpy()[:,0]
    return lam, {k:round(v,3) for k,v in est.items()}, round(float(np.sqrt(np.mean((X[f]-yg[f,0])**2))),3)
if __name__=='__main__':
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(run,[0.001,0.01,0.1,1.0]): print(r, flush=True)
