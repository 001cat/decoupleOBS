import obspy,sys,os,glob
import numpy as np
import pyfftw
from scipy.signal import spectrogram
from numpy import linalg as LA
from obspy import UTCDateTime
import matplotlib.pyplot as plt 

def trIntegrate(trIn):
    from scipy.fftpack import diff
    trOut = trIn.copy()
    trOut.data = diff(trOut.data,-1,trOut.stats.npts*trOut.stats.delta)
    return trOut

def trDifferentiate(trIn):
    from scipy.fftpack import diff
    trOut = trIn.copy()
    trOut.data = diff(trOut.data,1,trOut.stats.npts*trOut.stats.delta)
    return trOut

def doResample(st,dt,npts=None):
    st.filter('lowpass',corners=6,freq=0.5/dt/2,zerophase=True)
    starttime = st[0].stats.starttime
    endtime = st[0].stats.endtime
    for tr in st:
        if tr.stats.starttime > starttime:
            starttime = tr.stats.starttime
        if tr.stats.endtime < endtime:
            endtime = tr.stats.endtime
    if npts is None:
        npts = int(round((endtime - starttime)/dt)) - 5
    st.interpolate(1.0/dt,starttime=starttime,npts=npts)
    return st

def nextpow2(L):
    N = 1
    while N < L:
        N *= 2
    return N

def flatHann(N=256,taper=0.3):
    """ Flat Hanning window, taper at both sides 

    Args:
        N:      length of flat hanning window
        taper:  normalized length to be tapered at each side
    
    Returns:
        flat hanning window
    """
    win = np.ones(N)
    ntaper = int(N*taper)
    i = np.arange(2*ntaper)
    winHann = 0.5-0.5*np.cos(2*np.pi*i/(len(i)-1))
    win[:ntaper] = winHann[:ntaper]
    win[-ntaper:] = winHann[-ntaper:]
    return win

def smooth(a,N=5):
    ''' smooth function, similar to smooth function smooth(y,span) in Matlab(R2018a) '''
    if N%2 != 1:
        N = N-1
    b = np.convolve(a, np.ones(N)/N, mode='same')
    for i in range((N-1)//2):
        b[i]        = a[:(2*i+1)].mean()
        b[-1-i]     = a[(-1-2*i):].mean()
    return b
    
def ftest(res1,parms1,res2,parms2):
    ''' Function to perform a formal F-test
        function to perform a formal F-test on two sets of residuals of fits to
        data (res1, res2) that have respectively been fitted using parms1,parms2
        (where these are integers equal to the number of parameters used in each
        fit)
        IMPORTANTLY: 
        This will test whether the second fit (yielding res2) is statistically
        superior to the first fit, but used more parms. I.e.:
                sum(abs(res2)) < sum(abs(res1)) 
        and            parms2 > parms1

        the degrees of freedom for each fit are therefore:
        1) N - parms1
        2) N - parms2
        where N is the number of data, or length(res*)
        The residuals are just equal to dobs-dpred, so we square and sum these to
        get the chi^2 values (Ea)
        by Z. Eilon, python by Ayu Tsukimaya
    '''
    from scipy.stats import f
    res1 = np.array(res1)
    res2 = np.array(res2)
    N1, N2 = len(res1), len(res2)
    v1, v2 = N1-parms1, N2-parms2 # degrees of freedom
    Ea_1, Ea_2 = (res1*res1).sum(), (res2*res2).sum() # calculate chi^2 sums

    if Ea_1 == 0 or Ea_2 == 0 or v1 == 0 or v2 == 0:
        return 1
    else:
        Fobs = (Ea_1/v1)/(Ea_2/v2)
        return 1 - ( f.cdf(Fobs,v1,v2) - f.cdf(1/Fobs,v1,v2) )

def QC_staspectra_windows(trZ,tr1,tr2,trP,segT,overlap,passband,thres=1.5,saveFigs=False,
                          figDir=None):
    """ Function to identify time segments that potential transient signal exists
    Args:
        trZ,tr1,tr2,trP: traces of Z, two horizontal and P component
        segT:            segment(window) length, in seconds
        overlap:         normalized (1 = segT) overlap length of segments
        passband:        frequency band used in this function
        thres:           segment > STD*thres is suspect including transient signal
        saveFigs:        if you want to save new spectrogram and spectrum figures
        figDir:          directory to save figures
    Returns:
        winBs,winEs:    index of start and end point of each segment
        goodwins:       index of good segments
    """
    spectra = []
    spectraPlot = []
    for tr in [trZ,tr1,tr2,trP]:
        npts = round(segT/tr.stats.delta)
        f,t,sxx = spectrogram(tr.data, fs=1.0/tr.stats.delta, 
                    window=flatHann(N=npts,taper=min(0.3,200/segT)),
                    # window: Helen use 0.3, Ye use min(0.4,200/segT)
                    nperseg=npts, noverlap=int(npts*0.3), nfft=npts,mode='complex')
        sxx[sxx==0] = np.finfo(float).eps
        sxx = np.log10(abs(sxx))
        if saveFigs:
            spectraPlot.append(sxx.copy())
        sxx[(f<passband[0]) + (f>passband[1]) > 0,:] = 0
        for i in range(sxx.shape[1]):
            sxx[:,i] = smooth(sxx[:,i],N=50)
        for i in range(sxx.shape[0]):
            sxx[i,:] = sxx[i,:] - sxx[i,:].mean() 
        spectra.append(sxx)     
    winBs = np.round((t - segT/2) / trZ.stats.delta).astype(int)
    winEs = winBs + npts-1
    winN  = len(t)

    goodwins = np.array(list(range(winN)))
    moveon = False
    while moveon == False:
        normStd = []
        for win in goodwins:
            tmpIndex = [True if i in goodwins else False for i in range(winN) ]
            tmpIndex[win] = False
            normStd.append( [LA.norm(spectra[i][:,tmpIndex].std(axis=1)) for i in range(len(spectra))] )
        penalty = -(normStd - np.median(normStd,axis=0)).sum(axis=1)
        kill = penalty>thres*penalty.std()
        if np.any(kill) == False:
            break
        trypenalty = penalty[~kill]

        if ftest(penalty,1,trypenalty,1) < 0.05: # test with 95% confidence interva
            goodwins = goodwins[~kill]
            moveon = False
        else:
            moveon = True # no reason to remove more windows

    if saveFigs and (figDir is not None):
        fig,axes = plt.subplots(5,1,sharex=True,figsize=[8,10.5])
        nmDict = ['Z','H1','H2','P']
        for i in range(4):
            axes[i].pcolormesh(t,f,spectraPlot[i],cmap='jet')
            axes[i].set_title(nmDict[i])
            axes[i].set_ylabel('Frequency (Hz)')
        goodwinsInd = [i in goodwins for i in range(winN)]
        axes[-1].plot(t[goodwinsInd],np.ones(len(goodwins)),'kp')
        outfig = f'{trZ.stats.network}.{trZ.stats.station}.{trZ.stats.starttime.strftime("%Y%m%d%H%M%S")}'
        os.makedirs(f'{figDir}/STATIONS_DAILY',exist_ok=True)
        plt.savefig(f'{figDir}/STATIONS_DAILY/{outfig}.spectrogram.png')
        plt.close(fig.number)
        
    return [winBs,winEs,goodwins]

def calSpectra(trZ,tr1,tr2,trP,dsetDir,segT=3600.0, overlap=0.3, minwin=10, minLen=75000, trType='disp',
        QCpassband=None, saveSpec=True, calrotation=False, saveFigs=False, verbose=True, prefilt=None):
    """ Calculate spectrums after transient signal segments are removed
    Args:
        trZ,tr1,tr2,trP: traces of Z, two horizontal and P component
        dsetDir:         directory to save all results got in this module
        segT:            segment(window) length, in seconds
        overlap:         normalized (1 = segT) overlap length of segments
        minwin:          minimum number of segments required to continue decoupling.
        calrotation:     calculate orientation or not
        saveSpec:        save spectrums or not
        saveFigs:        save figures or not
    Returns:
        specprop:        spectrums and related propties
    """
    if QCpassband is None:
        passband = [0.01, .2]  # frequency band to check transient signal
                                # Ye Tian's setting: [1.0/20,1.0/11]
                                # Helen's setting:[0.004, .2]
    else:
        passband = QCpassband
    tiltfreq = [.005, .035]     # frequency band to check orientation
    threshold= 1.5              # a threshold to check transient signal
    figDir = f'{dsetDir}/FIGURES'

    if type(trZ)==str:
        trZ,tr1,tr2,trP = [obspy.read(trZ)[0],obspy.read(tr1)[0],
                           obspy.read(tr2)[0],obspy.read(trP)[0]]
    if prefilt is not None:
        for tr in [trZ,tr1,tr2,trP]:
            tr.detrend('linear')
            tr.taper(0.1,max_length=500)
            tr.filter('bandpass',freqmin=prefilt[0],freqmax=prefilt[1],zerophase=True)

    if trType == 'disp':
        pass
    elif trType == 'vel':
        # trZ.integrate()           # obspy.integrate is different from integral with FFT
        # tr1.integrate()           # but similar when you apply detrend+taper+highpass filter.          
        # tr2.integrate() 
        trZ = trIntegrate(trZ)
        tr1 = trIntegrate(tr1)
        tr2 = trIntegrate(tr2)
    else:
        raise ValueError(f'wrong trType:{trType}')       

    # check traces
    for tr in [trZ,tr1,tr2,trP]:
        if tr.stats.sac.stel != trZ.stats.sac.stel:     # check elevation
            print('Different elevation found!')
            return False
        if tr.stats.starttime != trZ.stats.starttime:   # check starttime
            print('Different starttime found!')
            return False
        if tr.stats.delta != trZ.stats.delta:           # check sample rate
            print('Different sample rate found!')
            return False
        if tr.stats.npts != trZ.stats.npts:             # check npts
            print('Different npts found!')
            return False
        if tr.stats.npts*tr.stats.delta < minLen:      # check record length
            print('Record length is not long enough!')
            return False
        if np.all(tr.data == 0):                        # check if all-zero record
            print('All-zero record!')
            return False

    # select noise window
    [iptWinBs,iptWinEs,goodwins] = QC_staspectra_windows(trZ,tr1,tr2,trP,segT=segT,overlap=overlap,
                                            passband=passband,thres=threshold,saveFigs=saveFigs,
                                            figDir=figDir)
    if len(goodwins) < minwin:
        print(f'Not enough good data segments! {minwin} required but {len(goodwins)} got')
        return False
    else:
        if verbose:
            print(f'{len(goodwins)} good segments. Proceeding...')

    # calculate spectrum of all window
    Nwin = len(iptWinBs)
    goodwinsInd = np.array([i in goodwins for i in range(Nwin)])
    npts = iptWinEs[0]-iptWinBs[0]+1
    NFFT = nextpow2(npts)
    dt = trZ.stats.delta
    fs = 1.0/dt
    f =  fs/2*np.linspace(0,1,NFFT//2+1)
    specAll = [[],[],[],[]]
    for i in range(len(iptWinBs)):
        for j,tr in enumerate([trZ,tr1,tr2,trP]):
            data = tr.data[iptWinBs[i]:iptWinEs[i]+1]
            data = data*flatHann(N=len(data), taper=min(0.3,200/segT)) # Helen use 0.3, Ye use min(0.4,200/segT)
            data = data-data.mean()
            spectrum = np.fft.fft(data,NFFT) * tr.stats.delta
            spectrum = spectrum[:NFFT//2+1]
            specAll[j].append(spectrum)
    specZ,spec1,spec2,specP = specAll

    # calculate cross-correlations(CC) in frequency domain
    # get avg CC of noise window
    cZZ,c11,c22,cPP = [],[],[],[]
    c1Z,c2Z,cPZ,c1P,c2P,c12 = [],[],[],[],[],[]
    for i in range(len(iptWinBs)):
        cZZ.append((specZ[i]*specZ[i].conj()).real*2/(NFFT*dt))
        c11.append((spec1[i]*spec1[i].conj()).real*2/(NFFT*dt))
        c22.append((spec2[i]*spec2[i].conj()).real*2/(NFFT*dt))
        cPP.append((specP[i]*specP[i].conj()).real*2/(NFFT*dt))
        c1Z.append(spec1[i]*specZ[i].conj()*2/(NFFT*dt))
        c2Z.append(spec2[i]*specZ[i].conj()*2/(NFFT*dt))
        cPZ.append(specP[i]*specZ[i].conj()*2/(NFFT*dt))
        c1P.append(spec1[i]*specP[i].conj()*2/(NFFT*dt))
        c2P.append(spec2[i]*specP[i].conj()*2/(NFFT*dt))
        c12.append(spec1[i]*spec2[i].conj()*2/(NFFT*dt))
    power = {'cZZ':np.array(cZZ)[goodwinsInd].mean(axis=0),
            'c11':np.array(c11)[goodwinsInd].mean(axis=0),
            'c22':np.array(c22)[goodwinsInd].mean(axis=0),
            'cPP':np.array(cPP)[goodwinsInd].mean(axis=0)}
    cross = {'c1Z':np.array(c1Z)[goodwinsInd].mean(axis=0),
            'c2Z':np.array(c2Z)[goodwinsInd].mean(axis=0),
            'cPZ':np.array(cPZ)[goodwinsInd].mean(axis=0),
            'c1P':np.array(c1P)[goodwinsInd].mean(axis=0),
            'c2P':np.array(c2P)[goodwinsInd].mean(axis=0),
            'c12':np.array(c12)[goodwinsInd].mean(axis=0)}

    # set output
    specprop = {'power':power,'cross':cross}
    specprop['params'] = {'f':f, 'network':trZ.stats.network, 'station':trZ.stats.station, 
                        'elev': trZ.stats.sac.stel,
                        'goodwins':goodwins,'iptWinBs':iptWinBs,'iptWinEs':iptWinEs,
                        'delta':dt,'overlap':overlap,'NFFT':NFFT}
    if saveFigs:
        def plotSpecWin(ax,f,spec,title):
            maxf = max(f)
            N = len(spec)
            f = np.tile(f,(N,1))
            spec = np.array([smooth(spec[i],N=100) for i in range(N)])
            ax.loglog(f.T,spec.T,'r',lw=0.5)
            ax.loglog(f[goodwinsInd].T,spec[goodwinsInd].T,'k',lw=0.5)
            ax.set_xlim(1/250,maxf)
            ax.set_title(title)
            ax.set_ylabel('PSD (m^2/Hz)')
        def plotSpecAvg(ax,f,spec,title):
            maxf = max(f)
            spec = smooth(spec,N=100)
            ax.loglog(f,spec,'k',lw=1.0)
            ax.set_xlim(1/250,maxf)
            ax.set_title(title)
        fig,axes = plt.subplots(4,2,gridspec_kw={'hspace':0.4},figsize=[8,10.5])
        plotSpecWin(axes[0,0],f,cZZ,'Z - all windows')
        plotSpecWin(axes[1,0],f,c11,'H1 - all windows')
        plotSpecWin(axes[2,0],f,c22,'H2 - all windows')
        plotSpecWin(axes[3,0],f,cPP,'P - all windows')
        plotSpecAvg(axes[0,1],f,power['cZZ'],'Z - Daily Average')
        plotSpecAvg(axes[1,1],f,power['c11'],'H1 - Daily Average')
        plotSpecAvg(axes[2,1],f,power['c22'],'H2 - Daily Average')
        plotSpecAvg(axes[3,1],f,power['cPP'],'P - Daily Average')
        outfname = f'{trZ.stats.network}.{trZ.stats.station}.{trZ.stats.starttime.strftime("%Y%m%d%H%M%S")}'
        os.makedirs(f'{dsetDir}/FIGURES/STATIONS_DAILY',exist_ok=True)
        plt.savefig(f'{dsetDir}/FIGURES/STATIONS_DAILY/{outfname}.dailySpectrum.png')
        plt.close()

    if calrotation:
        def estimateRotation(angles):
            angles = angles/180.0*np.pi
            cohLst,phLst = [],[]
            for ang in angles:
                specH = [np.sin(ang)*spec2[i]+np.cos(ang)*spec1[i] for i in range(len(iptWinBs))]
                cHH,cHZ = [],[]
                for i in range(len(iptWinBs)):
                    if i not in goodwins:
                        continue
                    cHH.append(abs(specH[i]*specH[i]).conj()*2/(NFFT*dt))
                    cHZ.append(specH[i]*specZ[i].conj()*2/(NFFT*dt))
                cHHAvg = np.array(cHH).mean(axis=0)
                cHZAvg = np.array(cHZ).mean(axis=0)
                cZZAvg = power['cZZ']

                cohAvg = (abs(cHZAvg)*abs(cHZAvg))/(cHHAvg*cZZAvg)
                phAvg = 180/np.pi*np.arctan2(cHZAvg.imag,cHZAvg.real)
                adAvg = abs(cHZAvg)/cHHAvg

                ind = (f>=tiltfreq[0]) * (f<=tiltfreq[1])
                cohLst.append(cohAvg[ind].mean())
                phLst.append(abs(phAvg[ind]).mean())
            ind = np.abs(phLst) < 90
            ind = np.where(ind)[0][np.argmax(np.array(cohLst)[ind])]
            max_coh = cohLst[ind]
            max_ori = angles[ind]/np.pi*180
            return (max_coh,max_ori,phLst,cohLst)

        angles = np.arange(0,360,10)
        max_coh,max_ori,phLst,cohLst = estimateRotation(angles)
        angles = np.arange(max_ori-10,max_ori+11,1)
        max_coh,max_ori,_,_ = estimateRotation(angles)
        specprop['params']['rotor'] = max_ori
        specprop['params']['rotcoh'] = max_coh

        ang = max_ori
        specH = [np.sin(ang)*spec2[i]+np.cos(ang)*spec1[i] for i in range(len(iptWinBs))]
        cHH,cHZ,cHP = [],[],[]
        for i in range(len(iptWinBs)):
            if i not in goodwins:
                continue
            cHH.append(specH[i]*specH[i].conj()*2/(NFFT*dt))
            cHZ.append(specH[i]*specZ[i].conj()*2/(NFFT*dt))
            cHP.append(specH[i]*specP[i].conj()*2/(NFFT*dt))
        cHHAvg = np.array(cHH).mean(axis=0)
        cHZAvg = np.array(cHZ).mean(axis=0)
        cHPAvg = np.array(cHP).mean(axis=0)
        specprop['rotation'] = {'cHH':cHHAvg,'cHZ':cHZAvg,'cHP':cHPAvg}

    if saveSpec:
        staID = f'{trZ.stats.network}.{trZ.stats.station}'
        os.makedirs(f'{dsetDir}/SPECTRA/{staID}',exist_ok=True)
        outfname = f'{staID}.{trZ.stats.starttime.strftime("%Y%m%d%H%M%S")}.spectra.npz'
        np.savez_compressed(f'{dsetDir}/SPECTRA/{staID}/{outfname}',specprop=specprop)

    return specprop

def QC_cleanstaspectra_days(cZZs,c11s,c22s,cPPs,f,passband,thres=1.5):
    """ Second quality control for ensemble of spectrums:
    Args:
        cZZs,c11s,c22s,cPPs: list of self-correlation in frequency domain of Z,
                             two horizontal and P component
        f: frequencies
        passband: frequency band used to do quality control
        thres: spectrums > STD*thres is suspect as outliers
    Returns:
        isGood: list of bools to indicate which spectrums are good
    """
    N = len(cZZs)
    spec = np.array([cZZs,c11s,c22s,cPPs])
    spec[spec==0] = np.finfo(float).eps
    spec = np.log10(abs(spec))
    spec[ :,:,(f>passband[1]) + (f<passband[0]) > 0 ] = 0
    for i in range(spec.shape[0]):
        for j in range(spec.shape[1]):
            spec[i,j,:] = smooth(spec[i,j,:],50)
        spec[i,:,:] = spec[i,:,:] - spec[i,:,:].mean(axis=0)

    goodLst = np.arange(N)
    moveon = False
    while moveon == False:
        normStd = []
        for igood in goodLst:
            tmpisgood = [True if i in goodLst else False for i in range(N)]
            tmpisgood[igood] = False
            normStd.append( [LA.norm(spec[i][tmpisgood,:].std(axis=1)) for i in range(len(spec))] )
        normStd = np.array(normStd)
        penalty = -(normStd - np.median(normStd,axis=0)).sum(axis=1)
        kill = penalty>thres*penalty.std()
        if np.any(kill) == False:
            break
        trypenalty = penalty[~kill]

        if ftest(penalty,1,trypenalty,1) < 0.05: # test with 95% confidence interva
            goodLst = goodLst[~kill]
            moveon = False
        else:
            moveon = True # no reason to remove more windows

    isGood = np.array([True if i in goodLst else False for i in range(N)])
    return isGood

def cleanSpectra(dsetDir,staID,saveAvg=True,saveFigs=True,QCpassband=None):
    """ check which spectrums/days are outliers, and calculate a average spectrum
    Args:
        dsetDir: dataset directory to save results got in this module
        staID: {network name}.{station name}
        saveAvg: save average spectrum or not
        svaeFigs: savefigs or not
    Returns:
        specprop: average specturm and related properties
    """
    if QCpassband is None:
        passband = [0.01, .2]  # frequency band to check transient signal
                                # Ye Tian's setting: [1.0/20,1.0/11]
                                # Helen's setting:[0.004, .2]
    else:
        passband = QCpassband
    thres= 1.5 # a threshold to check transient signal

    specprops = []
    specfiles = glob.glob(f'{dsetDir}/SPECTRA/{staID}/*.spectra.npz')
    for specfile in specfiles:
        specprops.append(np.load(specfile,allow_pickle=True)['specprop'][()])
        if specprops[0]['params']['NFFT'] != specprops[-1]['params']['NFFT']:
            raise ValueError()

    network = specprops[0]['params']['network']
    station = specprops[0]['params']['station']
    elev    = specprops[0]['params']['elev']
    f       = specprops[0]['params']['f']
    N       = len(specprops)
    Nwins   = [len(spec['params']['goodwins']) for spec in specprops]
    cZZs    = np.array([spec['power']['cZZ'] for spec in specprops])
    c11s    = np.array([spec['power']['c11'] for spec in specprops])
    c22s    = np.array([spec['power']['c22'] for spec in specprops])
    cPPs    = np.array([spec['power']['cPP'] for spec in specprops])
    c1Zs    = np.array([spec['cross']['c1Z'] for spec in specprops])
    c2Zs    = np.array([spec['cross']['c2Z'] for spec in specprops])
    cPZs    = np.array([spec['cross']['cPZ'] for spec in specprops])
    c1Ps    = np.array([spec['cross']['c1P'] for spec in specprops])
    c2Ps    = np.array([spec['cross']['c2P'] for spec in specprops])
    c12s    = np.array([spec['cross']['c12'] for spec in specprops])

    isGood = QC_cleanstaspectra_days(cZZs,c11s,c22s,cPPs,f,passband=passband,thres=thres)

    avgLst = []
    stdLst = []
    for cs in [cZZs,c11s,c22s,cPPs,c1Zs,c2Zs,cPZs,c1Ps,c2Ps,c12s]:
        cs = np.array(cs)
        csAvg = np.average(cs,axis=0,weights=isGood*np.array(Nwins))
        csStd = np.average((cs-csAvg)*(cs-csAvg),axis=0,weights=isGood*np.array(Nwins))
        csStd = csStd/(isGood.sum()-1+0.001)*isGood.sum()
        # https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
        avgLst.append(csAvg)
        stdLst.append(csStd)
    power = {'cZZ':avgLst[0],'c11':avgLst[1],'c22':avgLst[2],'cPP':avgLst[3],
            'cZZ-std':stdLst[0],'c11-std':stdLst[1],'c22-std':stdLst[2],'cPP-std':stdLst[3]}
    cross = {'c1Z':avgLst[4],'c2Z':avgLst[5],'cPZ':avgLst[6],
            'c1P':avgLst[7],'c2P':avgLst[8],'c12':avgLst[9],
            'c1Z-std':stdLst[4],'c2Z-std':stdLst[5],'cPZ-std':stdLst[6],
            'c1P-std':stdLst[7],'c2P-std':stdLst[8],'c12-std':stdLst[9]}
    params = {'f':f,'network':network,'station':station,'elev':elev}
    specprop = {'params':params,'power':power,'cross':cross}

    if saveFigs:
        os.makedirs(f'{dsetDir}/FIGURES/STATIONS_NOISEPROP',exist_ok=True)
        maxpow = [np.array(cs)[:,:-1].max()*100 for cs in [cZZs,c11s,c22s,cPPs]] # avoid Nyquist Frequency
        minpow = [np.array(cs)[:,:-1].min()/10  for cs in [cZZs,c11s,c22s,cPPs]]
        # plot power spectrum
        fig,axes = plt.subplots(4,1,sharex=True,figsize=[8,10.5])
        for cs,title,i in zip([cZZs,c11s,c22s,cPPs],['Z','H1','H2','P'],range(4)):
            cs = np.array(cs)
            for j in range(len(cs)):
                cs[j] = smooth(cs[j],40)
            axes[i].loglog(np.tile(f,(len(cs),1)).T,cs.T,c=[0.5,0.5,0.5],lw=0.5)
            cs = cs[~isGood]
            axes[i].loglog(np.tile(f,(len(cs),1)).T,cs.T,c=[1,0,0],lw=0.5)
            axes[i].set_ylim(minpow[i],maxpow[i])
            axes[i].set_xlim(1e-4,max(f))
            axes[i].set_ylabel('Power (db)')
            axes[i].set_title(title)
        axes[-1].set_xlabel('Frequency (Hz)')
        plt.savefig(f'{dsetDir}/FIGURES/STATIONS_NOISEPROP/{staID}.spectrum.png')
        plt.close()
        # plot Coherence,Phase,admittance
        coh = [abs(c1Zs)**2/(c11s*cZZs),
            abs(c2Zs)**2/(c22s*cZZs),
            abs(cPZs)**2/(cPPs*cZZs),
            abs(c1Ps)**2/(c11s*cPPs),
            abs(c2Ps)**2/(c22s*cPPs),
            abs(c12s)**2/(c11s*c22s)]
        phase = [np.arctan2(c1Zs.imag,c1Zs.real)/np.pi*180.0,
                np.arctan2(c2Zs.imag,c2Zs.real)/np.pi*180.0,
                np.arctan2(cPZs.imag,cPZs.real)/np.pi*180.0,
                np.arctan2(c1Ps.imag,c1Ps.real)/np.pi*180.0,
                np.arctan2(c2Ps.imag,c2Ps.real)/np.pi*180.0,
                np.arctan2(c12s.imag,c12s.real)/np.pi*180.0]
        admit = [abs(c1Zs)/c11s,
                abs(c2Zs)/c22s,
                abs(cPZs)/cPPs,
                abs(c1Ps)/c11s,
                abs(c2Ps)/c22s,
                abs(c12s)/c11s]
        for i in range(6):
            for j in range(N):
                coh[i][j,:] = smooth(coh[i][j,:],40)
                admit[i][j,:] = smooth(admit[i][j,:],40)
        fig,axes = plt.subplots(6,1,sharex=True,figsize=[8,10.5])
        for title,i in zip(['1Z','2Z','PZ','1P','2P','12'],range(6)):
            cs = coh[i]
            axes[i].semilogx(np.tile(f,(len(cs),1)).T,cs.T,c=[0.5,0.5,0.5],lw=0.5)
            cs = cs[~isGood]
            axes[i].semilogx(np.tile(f,(len(cs),1)).T,cs.T,c=[1,0,0],lw=0.5)
            axes[i].set_ylim(0,1)
            axes[i].set_xlim(1e-4,max(f))
            axes[i].set_ylabel('Coherence')
            axes[i].set_title(f'{staID} Coherence: {title}')
        axes[-1].set_xlabel('Frequency (Hz)')
        plt.savefig(f'{dsetDir}/FIGURES/STATIONS_NOISEPROP/{staID}.coherence.png')
        plt.close()
        fig,axes = plt.subplots(6,1,sharex=True,figsize=[8,10.5])
        for title,i in zip(['1Z','2Z','PZ','1P','2P','12'],range(6)):
            cs = phase[i]
            axes[i].semilogx(np.tile(f,(len(cs),1)).T,cs.T,'o',c=[0.5,0.5,0.5],markersize=1)
            cs = cs[~isGood]
            axes[i].semilogx(np.tile(f,(len(cs),1)).T,cs.T,'o',c=[1,0,0],markersize=1)
            axes[i].set_ylim(-200,200)
            axes[i].set_xlim(1e-4,max(f))
            axes[i].set_ylabel('Phase')
            axes[i].set_title(f'{staID} Phase: {title}')
        axes[-1].set_xlabel('Frequency (Hz)')
        plt.savefig(f'{dsetDir}/FIGURES/STATIONS_NOISEPROP/{staID}.phase.png')
        plt.close()
        fig,axes = plt.subplots(6,1,sharex=True,figsize=[8,10.5])
        for title,i in zip(['1Z','2Z','PZ','1P','2P','12'],range(6)):
            cs = admit[i]
            axes[i].loglog(np.tile(f,(len(cs),1)).T,cs.T,c=[0.5,0.5,0.5],lw=0.5)
            cs = cs[~isGood]
            axes[i].loglog(np.tile(f,(len(cs),1)).T,cs.T,c=[1,0,0],lw=0.5)
            axes[i].set_xlim(1e-4,max(f))
            axes[i].set_ylabel('Admittance')
            axes[i].set_title(f'{staID} Admittance: {title}')
        axes[-1].set_xlabel('Frequency (Hz)')
        plt.savefig(f'{dsetDir}/FIGURES/STATIONS_NOISEPROP/{staID}.admittance.png')
        plt.close()
        
    if saveAvg:
        staID = f'{network}.{station}'
        os.makedirs(f'{dsetDir}/SPECTRA-AVG/{staID}',exist_ok=True)
        outfname = f'{staID}.spectraAvg.npz'
        np.savez_compressed(f'{dsetDir}/SPECTRA-AVG/{staID}/{outfname}',specprop=specprop)
        np.savetxt(f'{dsetDir}/SPECTRA-AVG/{staID}/isgood.txt',np.column_stack((specfiles,isGood)),fmt="%s")

    return specprop

def calTransfer(dsetDir,staID,tfLst=['ZP','Z1','Z2-1','ZP-21','ZH','ZP-H'],
                saveTransfer=True,overwrite=True,saveFigs=True,
                freqLimP=None,freqLimH=0.105,freqLimL=0.005):
    """ calculate transfer function and build basis function for decoupling
    Args:
        dsetDir: dataset directory to save results got in this module
        staID: {network name}.{station name}
        tfLst: list of transfer function you want to calculate
               e.g.: 'ZP-21' means transfer function from P to Z after remove
               component 1 and 2 successively
        saveTransfer: save transfer function or not
        saveFigs: save figures or not
    Saved in output
        tfLst: list of transfer function name, same as input
        tfs:   list of transfer function
        deNoiseBases: basis function prepaired for decoupling
        params: related parameters
    This function ......
    """
    if False:
        def fullTfLst(tfLst):
            newLst = []
            for tfStr in tfLst:
                tmp = tfStr.replace('-','')
                for i in range(len(tmp)-1):
                    newStr = tmp[0]+tmp[(-1-i):]
                    if len(newStr) > 2:
                        newStr = newStr[:2]+'-'+newStr[2:]
                    newLst.append(newStr)
            for i in range(len(newLst)-1,-1,-1):
                if newLst[i] in newLst[:i]:
                    del newLst[i]
            return newLst
        tfLst = fullTfLst(tfLst)
    if saveFigs:
        fig,axes = plt.subplots(len(tfLst),1,sharex=True,figsize=[8,10.5])
        isGoods = np.loadtxt(f'{dsetDir}/SPECTRA-AVG/{staID}/isgood.txt',dtype=str)
        if len(isGoods.shape) == 1:
            isGoods = [isGoods]
        isGoodDict = {}
        for isGood in isGoods:
            isGoodDict[isGood[0]] = isGood[1] == 'True'
    for specFile in glob.glob(f'{dsetDir}/SPECTRA/{staID}/*.npz'):
        os.makedirs(f'{dsetDir}/TRANSFUN/{staID}',exist_ok=True)
        outfname = f'{staID}.{specFile.split("/")[-1].split(".")[2]}.transfun.npz'
        outfpath = f'{dsetDir}/TRANSFUN/{staID}/{outfname}'
        if os.path.exists(outfpath):
            if overwrite:
                print(f'{outfpath} exists, overwrite it!')
            else:
                print(f'{outfpath} exists, skip it!')
                continue
        specprop = np.load(specFile,allow_pickle=True)['specprop'][()]
        ccDict = {'ZZ':specprop['power']['cZZ'],
                '11':specprop['power']['c11'],
                '22':specprop['power']['c22'],
                'PP':specprop['power']['cPP'],
                'HH':specprop['rotation']['cHH'],
                '1Z':specprop['cross']['c1Z'],
                '2Z':specprop['cross']['c2Z'],
                'PZ':specprop['cross']['cPZ'],
                'HZ':specprop['rotation']['cHZ'],
                '1P':specprop['cross']['c1P'],
                '2P':specprop['cross']['c2P'],
                'HP':specprop['rotation']['cHP'],
                '12':specprop['cross']['c12']}
        f = specprop['params']['f']
        dayID = specFile.split('/')[-1].split('.')[2]
        params = {'f':f,
                'staID':staID,
                'dayID':dayID,
                'elevation':specprop['params']['elev'],
                'oriAng':specprop['params']['rotor'],
                'NFFT':specprop['params']['NFFT'],
                'delta':specprop['params']['delta'],
                'nptsWin':specprop['params']['iptWinEs'][0] - specprop['params']['iptWinBs'][0] + 1}
        
        tfs = []
        deNoiseBases = []
        elev = -specprop['params']['elev']
        if freqLimP is None:
            freqLimP = np.sqrt(9.8/(2*np.pi*elev)) + 0.005
        def buildTaper(f,lo,hi):
            taper = np.ones(len(f))
            f1,f2,f3,f4 = lo*0.9,lo,hi,hi+0.01*max(f) #Helen
            # f1,f2,f3,f4 = lo-(hi-lo)/15.0,lo,hi,hi+(hi-lo)/15.0   # Ye
            taper[f<f1] = 0
            taper[f>f4] = 0
            f12 = f[(f>=f1)*(f<=f2)]
            taper[(f>=f1)*(f<=f2)] = np.cos(2*np.pi*(f12-f1)/(f2-f1))
            f34 = f[(f>=f3)*(f<=f4)]
            taper[(f>=f3)*(f<=f4)] = np.cos(2*np.pi*(f34-f3)/(f4-f3))
            return taper
        taperP = buildTaper(f,freqLimL,freqLimP)
        taperH = buildTaper(f,freqLimL,freqLimH)
        # freqLimL: low limit of frequency band pass when decouple
        # freqLimH: high limit for tilt
        # freqLimP: high limit for pressure/compliance
        def retriveCC(ccid, ccDict):
            if ccid in ccDict.keys():
                return ccDict[ccid]
            else:
                ccid = ccid[::-1]
                if ccid in ccDict.keys():
                    return ccDict[ccid].conj()
                else:
                    raise ValueError('Unknown cross-correlation ID!')
        def getTaper(comp1,comp2):
            compZ = ['Z']
            compH = ['1','2','H']
            compP = ['P']
            if (comp1 in compZ and comp2 in compH) or (comp2 in compZ and comp1 in compH):
                return taperH
            elif (comp1 in compH and comp2 in compH):
                return taperH
            elif (comp1 in compZ+compH and comp2 in compP) or (comp2 in compZ+compH and comp1 in compP):
                return taperP
            else:
                raise ValueError('Error in getTaper: unaccepted component pair')
        def getTF(tfcode,ccDict):
            tfcode = tfcode.replace('-','')
            if len(tfcode) == 4:
                z,c,b,a = tfcode
                taper1 = getTaper(c,b)+getTaper(z,b)-getTaper(c,b)*getTaper(z,b)
                taper2 = getTaper(c,b)*2-getTaper(c,b)**2
                x = getTF(z+c+a,ccDict) - getTF(z+b+a,ccDict)*getTF(b+c+a,ccDict)*taper1
                y = 1-getTF(c+b+a,ccDict)*getTF(b+c+a,ccDict)*taper2
                return x/y
            elif len(tfcode) == 3:
                z,b,a = tfcode
                taper1 = getTaper(b,a)+getTaper(z,a)-getTaper(b,a)*getTaper(z,a)
                taper2 = getTaper(b,a)*2-getTaper(b,a)**2
                x = getTF(z+b,ccDict) - getTF(z+a,ccDict)*getTF(a+b,ccDict)*taper1
                y = 1-getTF(a+b,ccDict)*getTF(b+a,ccDict)*taper2
                # if np.any(x==0) or np.any(y==0):
                #     print(tfcode)
                return x/y
            elif len(tfcode) == 2:
                return retriveCC(tfcode,ccDict) / retriveCC(tfcode[1]*2,ccDict)
            else:
                raise ValueError('Not supported yet!')
        def getdeNoiseBase(tfcode,ccDict):
            tfcode = tfcode.replace('-','')
            tfFunc = getTF(tfcode,ccDict)
            taper  = getTaper(tfcode[0],tfcode[1])
            compList = ['Z','1','2','P','H']
            if len(tfcode) == 4:
                z,c,b,a = tfcode
                return getdeNoiseBase(z+b+a,ccDict) - taper*tfFunc*getdeNoiseBase(c+b+a,ccDict)
            if len(tfcode) == 3:
                z,b,a = tfcode
                return getdeNoiseBase(z+a,ccDict) - taper*tfFunc*getdeNoiseBase(b+a,ccDict)
            if len(tfcode) == 2:
                base = np.zeros( [5,len(tfFunc)], dtype='complex')
                base[compList.index(tfcode[0])] = 1
                base[compList.index(tfcode[1])] = -taper*tfFunc
                return base
        for tfcode in tfLst:
            tfs.append(getTF(tfcode,ccDict))
            deNoiseBases.append(getdeNoiseBase(tfcode,ccDict))
        if saveTransfer:
            os.makedirs(f'{dsetDir}/TRANSFUN/{staID}',exist_ok=True)
            np.savez_compressed(f'{dsetDir}/TRANSFUN/{staID}/{outfname}',
                                tfLst=tfLst,tfs=tfs,deNoiseBases=deNoiseBases,params=params)

        if saveFigs:
            if isGoodDict[specFile]:
                c = [0.5,0.5,0.5]
            else:
                c = [1,0,0]
            for i in range(len(tfLst)):
                axes[i].loglog(f,abs(tfs[i]),c=c,lw=0.5)
                axes[i].set_xlim(1e-4,max(f))
                # axes[i].set_ylim(ylim[i][0],ylim[i][1])
                axes[i].set_title(f'{staID} Transfer Function {tfLst[i]}')
    os.makedirs(f'{dsetDir}/FIGURES/STATIONS_TRANSFUNC',exist_ok=True)
    plt.savefig(f'{dsetDir}/FIGURES/STATIONS_TRANSFUNC/{staID}.transfunc.png')
    plt.close()

def deCouple(trZ,tr1,tr2,trP,dsetDir,staID,evtID,evtTime,saveTraces=True,saveFigs=True,
    trType='disp',prefilt=None):
    """ decouple component based on previous calculated basis functions
    Args:
        trZ,tr1,tr2,trP: traces of Z, two horizontal and P component
        dsetDir:         directory to save all results got in this module
        staID: {network name}.{station name}
        evtID: event id, used to save decoupled records
        evtTime: event origin time, used to find basis functions from a day near it
        saveTraces: save all decoupled records or not
        saveFigs: save figures or not
    Returns:
        trCorrected: list of traces decoupled
    """
    # load event record
    if type(trZ)==str:
        trZ,tr1,tr2,trP = [obspy.read(trZ)[0],obspy.read(tr1)[0],
                           obspy.read(tr2)[0],obspy.read(trP)[0]]

    if prefilt is not None:
        for tr in [trZ,tr1,tr2,trP]:
            tr.detrend('linear')
            tr.taper(0.1,max_length=500)
            tr.filter('bandpass',freqmin=prefilt[0],freqmax=prefilt[1],zerophase=True)

    if trType == 'disp':
        pass
    elif trType == 'vel':
        # trZ.integrate()
        # tr1.integrate()
        # tr2.integrate()
        trZ = trIntegrate(trZ)
        tr1 = trIntegrate(tr1)
        tr2 = trIntegrate(tr2)
    else:
        raise ValueError(f'wrong trType:{trType}')   
    # check traces
    for tr in [trZ,tr1,tr2,trP]:
        if tr.stats.starttime != trZ.stats.starttime:   # check starttime
            print('Different starttime found!')
            return False
        if tr.stats.delta != trZ.stats.delta:           # check sample rate
            print('Different sample rate found!')
            return False
        if tr.stats.npts != trZ.stats.npts:             # check npts
            print('Different npts found!')
            return False
        if np.all(tr.data == 0):                        # check if all-zero record
            print('All-zero record!')
            return False

    trZOri = trZ.copy()

    # get property of event record
    dt      = trZ.stats.delta
    npts    = trZ.stats.npts
    NFFT    = nextpow2(npts)
    fs      = 1.0/dt
    f       = fs/2*np.linspace(0,1,NFFT//2+1)

    # calculate specs, add zeros at both ends before fft
    specs = []
    npad0 = (NFFT - npts)//2
    for tr in [trZ,tr1,tr2,trP]:
        data = tr.data
        data = data * flatHann(len(data),0.075)
        data = data - data.mean()
        datapad = np.zeros(NFFT)
        datapad[npad0:npad0+npts] = data
        spec = np.fft.fft(datapad,NFFT)*dt
        spec = spec[:NFFT//2+1]
        specs.append(spec)

    # find noise record, closest to event and prefer noise record before event
    evtTime = UTCDateTime(evtTime)
    transFiles = glob.glob(f'{dsetDir}/TRANSFUN/{staID}/*.transfun.npz')
    transDates = [transFile.split('/')[-1].split('.')[2] for transFile in transFiles]
    tmp = abs(np.array([UTCDateTime(i)+43200-evtTime for i in transDates]))
    transFile = transFiles[np.argmin(tmp)]

    # load from transfer estimates
    tmp = np.load(transFile,allow_pickle=True)
    tfLst = list(tmp['tfLst'][()])
    deNoiseBases = tmp['deNoiseBases'][()]
    paramsBases = tmp['params'][()]
    fBases = paramsBases['f']

    # calculate H(rotated) component
    ang = paramsBases['oriAng']/180.0*np.pi
    specs.append( np.sin(ang)*specs[2]+np.cos(ang)*specs[1] )
    specs = np.array(specs)

    # interpolate to array f of event record
    deNoisedSpecs = []
    for deNoiseBase,tfCode in zip(deNoiseBases,tfLst):
        deNoiseBase = [np.interp(f,fBases,coef) for coef in deNoiseBase]
        spec = (specs*deNoiseBase).sum(axis=0)
        deNoisedSpecs.append(spec)

    trCorrected = []
    for deNoisedSpec in deNoisedSpecs:
        data = np.fft.ifft(2*deNoisedSpec,NFFT).real/dt
        tr = trZ.copy()
        tr.data = data[npad0:npad0+npts]
        trCorrected.append(tr)
    
    if saveTraces:
        for tr,tfCode in zip(trCorrected,tfLst):
            os.makedirs(f'{dsetDir}/CORRECTED/{staID}',exist_ok=True)
            tr.write(f'{dsetDir}/CORRECTED/{staID}/{staID}.{evtID}.{tfCode}.SAC',format='SAC')
        trZOri.write(f'{dsetDir}/CORRECTED/{staID}/{staID}.{evtID}.ori.SAC',format='SAC')

    if saveFigs:
        os.makedirs(f'{dsetDir}/FIGURES/CORRECTED',exist_ok=True)
        trLst = [trZ,tr1,tr2,trP]
        fig,axes = plt.subplots(len(trLst),1,sharex=True,figsize=[8,10.5])
        for i in range(len(trLst)):
            tr = trLst[i].copy()
            tr.filter('bandpass',freqmin=1.0/200.0,freqmax=1.0/10.0,corners=2,zerophase=True)
            axes[i].plot(tr.times(),tr.data,'k',lw=0.5)
            axes[i].set_xlim(tr.times()[0],tr.times()[-1])
            axes[i].set_title(('Z','H1','H2','P')[i])
        plt.savefig(f'{dsetDir}/FIGURES/CORRECTED/{staID}.{evtID}.origin.png')
        plt.close()

        fig,axes = plt.subplots(len(trCorrected),1,sharex=True,figsize=[8,10.5])
        for i in range(len(trCorrected)):
            tr = trCorrected[i].copy()
            tr.filter('bandpass',freqmin=1.0/200.0,freqmax=1.0/10.0,corners=2,zerophase=True)
            axes[i].plot(tr.times(),tr.data,'k',lw=0.5)
            axes[i].set_xlim(tr.times()[0],tr.times()[-1])
            axes[i].set_title(tfLst[i])
        plt.savefig(f'{dsetDir}/FIGURES/CORRECTED/{staID}.{evtID}.corrected.png')
        plt.close()

    return trCorrected




if __name__ == "__main__":
    plt.ioff()
    
    net = '7D'
    # sta = 'G30A'
    # chans = 'BHZ,BH1,BH2,BDH'
    sta = 'FN07A'
    chans = 'HHZ,HH1,HH2,HDH'

    from obspy.clients.fdsn import Client
    from obspy.core.util.attribdict import AttribDict
    def addSacHeader(st,inv):
        for tr in st:
            seedID = f'{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}'
            tr.stats.sac = AttribDict()
            staCoord = inv.get_coordinates(seedID,tr.stats.starttime)
            tr.stats.sac.stel = staCoord['elevation']
            tr.stats.sac.stla = staCoord['latitude']
            tr.stats.sac.stlo = staCoord['longitude']
    
    client = Client('IRIS')
    inv = client.get_stations(network=net,station=sta,location='*',channel=chans,
                            level='channel')
    cata = client.get_events(starttime='2012-02-02',endtime='2012-02-03',minmagnitude=6.0)
    evtTime = cata[0].preferred_origin().time

    print('Downloading Noise ... ')
    stNoise = client.get_waveforms(network=net,station=sta,location='*',channel=chans,
                                starttime='2012-02-01',endtime='2012-02-02',attach_response=True)

    print('Downloading Event ... ')
    stEvent = client.get_waveforms(network=net,station=sta,location='*',channel=chans,
                                starttime=evtTime-1200,endtime=evtTime+7100,attach_response=True)

    print('Removing response ... ')
    for tr in stNoise+stEvent:
        if tr.stats.response.response_stages[0].input_units == 'PA':
            tr.remove_response(pre_filt=[0.001,0.002,1000,2000])
        elif tr.stats.response.response_stages[0].input_units in ('M/S','M'):
            tr.remove_response(pre_filt=[0.001,0.002,1000,2000], output='DISP')
        else:
            print(f'Unknown input unit, no response removed: {tr.get_id()}')

    print('Resampling ... ')
    stNoise = doResample(stNoise.copy(),0.2)
    stEvent = doResample(stEvent.copy(),0.2,round(7200.0/0.2))

    addSacHeader(stNoise,inv)

    print('Calculating Spectra ... ')
    specprop = calSpectra(stNoise[3],stNoise[1],stNoise[2],stNoise[0],dsetDir='NOISETC',
                        calrotation=True,segT=3600.0,saveFigs=True,saveSpec=True)

    print('Calculating Spectra - Average ... ')
    specpropAvg = cleanSpectra(dsetDir='NOISETC',staID=f'{net}.{sta}')

    print('Calculating Transfer Function ... ')
    calTransfer(dsetDir='NOISETC',staID=f'{net}.{sta}')

    print('Correcting ... ')
    trCorrected = deCouple(stEvent[3],stEvent[1],stEvent[2],stEvent[0],dsetDir='NOISETC',
                        staID=f'{net}.{sta}',evtID='201202021330',evtTime=evtTime)












