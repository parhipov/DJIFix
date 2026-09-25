import numpy as np, json, sys
from scipy.signal import butter, sosfiltfilt
sys.path.insert(0,'K:/Work/Python/DJI fix/src')
from rollfix import logv, qmul, qconj
def incs(q):
    q=np.asarray(q,float); s=np.sign(np.sum(q[:-1]*q[1:],1)); s[s==0]=1
    return logv(qmul(qconj(q[:-1]),q[1:]*s[:,None]))
for clip in ('o4pro_13-17s','o4lite_53-58s'):
    print('==',clip)
    for v in ('orig','00_control','telemetry_fixed'):
        cam=json.load(open('%s_%s_cam.json'%(clip,v))); cam=sorted(cam,key=lambda c:c['frame'])
        Sq=incs([c['stab_quat'] for c in cam]); Oq=incs([c['org_quat'] for c in cam])
        z=np.load('%s_%s_stab_resid.npz'%(clip,v)); r=z['r'].copy(); fps=float(z['fps']); W=float(z['W'])
        r[:,2]=np.radians(r[:,2])*W/2; r=np.nan_to_num(r)
        n=min(len(r),len(Sq)); r=r[:n]; Sq=Sq[:n]
        # which path axis explains which render axis: fit on 0.3-3 Hz
        sos=butter(2,[0.3,3],btype='band',fs=fps,output='sos')
        X=sosfiltfilt(sos,np.cumsum(Sq,0),axis=0)[10:-10]; Y=sosfiltfilt(sos,np.cumsum(r,0),axis=0)[10:-10]
        M,*_=np.linalg.lstsq(X,Y,rcond=None)
        if v=='orig': print('  map path(deg)->render(px):\n',np.round(M.T,1)); M0=M
        pred=Sq@M0; u=r-pred
        row=[]
        for lo,hi in ((0.3,1),(1,4),(4,8),(8,24.5)):
            s2=butter(2,[lo,hi],btype='band',fs=fps,output='sos')
            B=lambda x: sosfiltfilt(s2,np.cumsum(x,0),axis=0)[10:-10].std(0)
            row.append((B(r),B(pred),B(u)))
        print('  %-16s'%v+'  '.join('%g-%gHz tot %s path %s ERR %s'%(b[0],b[1],' '.join('%.1f'%x for x in t),' '.join('%.1f'%x for x in p),' '.join('%.2f'%x for x in e)) for b,(t,p,e) in zip(((0.3,1),(1,4),(4,8),(8,25)),row)))
