import numpy as np, sys
from scipy.signal import butter, sosfiltfilt
for clip in ('o4pro_13-17s','o4lite_53-58s'):
    print('==', clip, ' rms of residual POSITION in px (x, y, roll at frame edge), per band; and per-frame jitter px')
    print('%-16s %-22s %-22s %-22s %-22s %s'%('version','1-4 Hz','4-8 Hz','8-25 Hz','per-frame 2nd diff',''))
    for v in ('orig','00_control','telemetry_fixed','B_notiming','C_notiming_noev','D_roll_spikes'):
        z=np.load('%s_%s_stab_resid.npz'%(clip,v)); r=z['r'].copy(); fps=float(z['fps']); W=float(z['W'])
        r[:,2]=np.radians(r[:,2])*W/2
        r=np.where(np.isfinite(r),r,0)
        pos=np.cumsum(r,0)
        row=[]
        for lo,hi in ((1,4),(4,8),(8,24.5)):
            sos=butter(2,[lo,hi],btype='band',fs=fps,output='sos'); row.append(sosfiltfilt(sos,pos,axis=0)[10:-10].std(0))
        d2=np.diff(r,axis=0)[5:-5]
        row.append(np.sqrt((d2**2).mean(0))/np.sqrt(2))
        print('%-16s '%v+' '.join('%-22s'%(' '.join('%6.2f'%x for x in q)) for q in row))
