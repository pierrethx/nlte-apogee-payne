#import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
import os, sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
#from astroNN.apogee import wavelength_solution
from scipy.signal import convolve
from scipy.stats import norm
#import emcee,corner

from fit_one import *

def load_spectra_i(filt,nn,lims,totaltasks,taskid):
    # read in list of spectra locations
    fils=np.loadtxt(filt,dtype=str)
    # split up, if parallel process
    fils = fils[taskid::totaltasks]    

    waves=[]
    fluxs=[]
    errrs=[]
    masks=[]

    #using summary file to make this quicker
    #t=Table.read("astraAllStarASPCAP-0.6.0.fits.gz",hdu=2)
    t=Table.read("./trunc_raw_astra_0.6.0.fits.gz",hdu=1)

    oo=[]
    # looping through spectra in the chunk
    for fil in fils:
        # load in spectra
        wave, flux, errs, model, mask, h = load_mwmStar(fil )
        wave,flux,errs,mask=chunkup(wave,flux,errs,mask)
        # append sdss_id, which will be used to identify the star
        oo.append(int(os.path.basename(fil[:-5]).split("-")[-1]) )
        # regularizing errors
        for i in range(len(wave)):
            errs[i][np.isinf(errs[i])|np.isnan(errs[i])]=999999.
            flux[i][np.isinf(flux[i])|np.isnan(flux[i])]=0.
            #errs[flux>1.01]=999999.
            errs[i][mask[i]!=0]=999999.
        waves.append(wave)
        fluxs.append(flux)
        errrs.append(errs)
        masks.append(mask)


    # the labels of the neural network, hardcoded
    adjss=[0, 0, 0, 0.05, 0.17, 0.2, 0.02, 0.06, 0.13, 0.12, 0.06, 0.01, 0.04, 0.08] # grevesse->magg solar scale
    plist=['teff','logg','v_micro','fe_h','c_h','n_h','mg_h','al_h','mn_h','na_h','ca_h','ni_h','ti_h','si_h'] 
    plist=[f"raw_{x}" for x in plist]
    params = np.atleast_2d([list(t[t['sdss_id'] == sid][plist][0].as_void()) for sid in oo])    
    print("params",params)
    # Apply abundance scale corrections
    for i in [3,4,5,6,7,8,9,10,11,12,13]:
        params[:,i]-=adjss[i]
    # Astra parameters are [X/H], we want the labels as [X/Fe]
    for i in [4,5,6,7,8,9,10,11,12,13]:
        params[:,i]-=params[:,3]
    # and also load in the flags
    flagz=["spectrum_flags","fe_h_flags","c_h_flags","n_h_flags","mg_h_flags","al_h_flags","mn_h_flags","na_h_flags","ca_h_flags","ni_h_flags","ti_h_flags","si_h_flags"]
    flagl=[list(t[t['sdss_id'] == sid][flagz][0].as_void()) for sid in oo]

    # the fitting occurs in normalized label space. We use Astra params as fitting seed 
    # Labels exceeding the training limits will break the code, but ideally, we only fit spectra that are thought to be covered by the NN
    s_labelz=normalize_labels(params,nn)
    s_labelz[s_labelz<-0.5]=-0.5
    s_labelz[s_labelz>0.5]=0.5
    print("norm p0",s_labelz)
    return waves, s_labelz, fluxs, errrs, oo,flagl 

def chunkup(wave,spect,*args):
    assert len(wave)==len(spect)

    argls=[]
    for arg in args:
        assert len(wave)==len(arg)
        argls.append([])
    
    lambs=[]
    norms=[]
    chunki=-1
    inchunk=False
    for w in range(len(wave)):
        if chunki==-1:
            if np.isnan(spect[w]) or np.isinf(spect[w]):
                pass
            else:
                chunki=w
                inchunk=True
        else:
            if (np.isnan(spect[w]) or np.isinf(spect[w])) and inchunk:
                lambs.append(wave[chunki:w-1])
                norms.append(spect[chunki:w-1])
                for a in range(len(args)):
                    argls[a].append(args[a][chunki:w-1])
                chunki=-1
                inchunk=False
            else:
                pass
        #print(chunki,spect[w],end=" ")
    return lambs,norms,*argls

def gen_predict(dic,nlabels,NN_red,wavel,wave,mask=None,maskw=None):
    # wavel is the wavelengths of the neural network
    # wave is the wavelengths of the data
    
    #lets assume that wave is a list of 1d arrays and that we can concat it somewhere in here
    #print(nlabels.shape,NN_red[0].shape)
    
    nparams = NN_red[0].shape[-1]
    #len of nlabels should be nparams+nchunks*order+2 (for rv and broadening)
    nchunks=len(wave)
    assert (len(nlabels)-2-nparams)%nchunks==0
    ncoeffs=int((len(nlabels)-2-nparams)/nchunks)
    #cpolyl = [np.poly1d(nlabels[nparams+q*ncoeffs:nparams+(q+1)*ncoeffs]) for q in range(nchunks)]
    cpolyl = [np.polynomial.chebyshev.Chebyshev(nlabels[nparams+q*ncoeffs:nparams+(q+1)*ncoeffs]) for q in range(nchunks)]

    # at given nlabels, generate spectrum from emulator
    spec_predict=get_spectrum_from_neural_net(dic,nlabels[:nparams], NN_red)
    # convolve with broadening parameter
    spec_predict=convo(spec_predict,nlabels[-1],wavel)
    # doppler shift using rv parameter, then resample onto original wavelength grid
    vc=nlabels[-2]/3e5
    wavel2=wavel*np.sqrt((1+vc)/(1-vc))
    spec_predict=np.interp(wavel,wavel2,spec_predict)
    predict=interp1d(wavel2,spec_predict)
    # if we were working in a really sparse sampling the order and implementation of these would be really important
    # but generally, our NNE is trained at a higher resolution than the data
    # ~~~~
    
    norm_predict=[]
    errs_predict=[]
    
    for l,lam in enumerate(wave):
        # each is a chip
        norm_pred=predict(lam)
        
        # apply continuum normalization
        #print()
        norm_pred*=(1+cpolyl[l]( (lam-np.median(lam))/np.ptp(lam) ))
        
        # populate error accross spectrum. We let all pixels in model be equal except where masked
        errs_pred=np.ones_like(norm_pred)*0.001
        if not mask is None:
            if maskw is None:
                mask2=np.interp(lam,wavel,mask)>0
            else:
                mask2=np.interp(lam,maskw,mask)>0
            errs_pred[mask2] = 99999.
        
        norm_predict.append(norm_pred)
        errs_predict.append(errs_pred)
    assert len(wave)==len(norm_predict)

    return norm_predict,errs_predict

def fit_func(nlabels,NN_red,normspec,errsspec,lambd,amount,dic,wavel,mask=None,maskw=None):
    #wave_predict=np.concatenate(lambd)
    norm_predict,errs_predict=gen_predict(dic,nlabels,NN_red,wavel,lambd,mask,maskw)

    errs = np.sqrt( np.concatenate(errs_predict)**2 + np.concatenate(errsspec)**2)
    errs[(~np.isfinite(errs)) | (errs > 300) | (errs<0)] =999.
    resid = (np.concatenate(normspec)-np.concatenate(norm_predict))/errs
    return resid

def fit_one_ls(NN_red,wave,dic,normspec,errsspec,lambd,p0,mask=None,maskw=None):
    npar=(NN_red[0].shape[-1])
    bound0=np.full(len(p0),-0.5)
    bound1=np.full(len(p0),0.5)
    
    bound0[npar:-2]=-10
    bound1[npar:-2]=10
    
    bound0[-2]=-200
    bound0[-1]=0
    bound1[-2]=200
    bound1[-1]=5

    amount=16000/22500

    tol = 1e-12#5e-5

    ar=(NN_red,normspec,errsspec,lambd,amount,dic,wave,mask,maskw)
    try:
        res = least_squares(fit_func, p0,ftol=tol, xtol=tol, method='trf',bounds=(bound0,bound1),args=ar)
    except ValueError:
        print(bound0,bound1)
        assert False
    popt = res.x
    print("fitfunc",fit_func(popt,*ar))

    p1=np.copy(popt)
    p1[:npar]=popt[:npar]

    chi_i= np.sum( fit_func(p1,*ar)**2)
    chi_f=np.sum( fit_func(popt,*ar)**2)
    print("chis",chi_i,chi_f)
    return popt,chi_i,chi_f

def write_to_all_a(direct,tag,st,ii,chi_i,chi_f,true_lab,fit_lab,flagl):
    assert (len(chi_i)==len(chi_f)) and (len(true_lab)==len(fit_lab))
    pafz=os.path.join(direct,f"rcparamfit{st}_{tag}.txt")
    print(pafz)
    with open(pafz,'a') as mf:
        for q in range(len(chi_i)):
            true_string=" ".join([f"{la:.4f}" for la in true_lab[q]])
            fit_string=" ".join([f"{la:.4f}" for la in fit_lab[q]])
            flag_string=" ".join([f"{int(la)}" for la in flagl[q]])
            mf.write(f"{ii[q]} {chi_i[q]:.4f} {true_string}  {chi_f[q]:.4f} {fit_string} {flag_string}\n")

if __name__=="__main__":
    s0=sys.argv[1] # spectra data (list of spectra addresses)
    s1=sys.argv[2] # inputtrain file (contains reference to coefficients and info about the training)

    s15=sys.argv[3] # mask file
    try:
        s3=int(sys.argv[4]) #number of parallel jobs
        s4=int(sys.argv[5]) #index of parallel jobs
    except:
        # no parallel
        s3,s4=1,0

    dic,inpu,x,lims=(parse_input_file(s1,0))
    print(dic,inpu)

    # load in NN and all data to fit
    nnpath=os.path.join(".",dic["direct"],"NN_normalized_spectra"+dic["tag"]+".npz") 
    theNN=load_NN(nnpath)
    wav,s_lab,spec,errs,ids,fgz=load_spectra_i(s0,theNN,lims,s3,s4)
       
    # load in data fitting mask and interpolate onto relevant grid (hardcoded, this is the one we trained on)
    trwav = np.linspace(15000,17000,20001)
    try:
        mask0=np.loadtxt(s15).T
        mask = np.interp(trwav,mask0[0],mask0[1])
    except:
        mask=None

    s=1e4/trwav
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    trwav =trwav*n
    # synthesis is defaultly in air so we convert it to vacuum to match the observations (which are in vacuum) 
 
    ci_s=[]
    cf_s=[]
    p0_s=[]
    pf_s=[] 
    ii_s=[]
    
    #contmask=np.load("dr17_contmask.npy")
    #cwav = 10.0 ** np.arange(4.179, 4.179 + 8575 * 6.0 * 10.0**-6.0, 6.0 * 10.0**-6.0)
    #cwav=np.concatenate((cwav[246:3274],cwav[3585:6080],cwav[6344:8335]))

    for s in range(len(spec)):
        print(s)
        if True:
            #lam,norma,err=resample(wav,spec[s])
            lam=wav[s]
            norma=spec[s]
            err=errs[s]

            #a.plot(wav[s],errs[s],color='r')
            #plt.show()
            
            poly0=[]
            order=3
            '''
            for i in range(len(lam)):
                cmask=np.ceil(np.interp(lam[i],cwav,contmask)).astype(bool)
                thresh=0.95
                rnorm=norma[i][cmask]
                rwave=((lam[i]-np.median(lam[i]))/np.ptp(lam[i]))[cmask][rnorm>thresh]
                rerr =err[i][cmask][rnorm>thresh]
                rnorm=rnorm[rnorm>thresh]-1
                poog2 = np.polynomial.chebyshev.Chebyshev.fit((rwave),(rnorm),deg=order,w=1/(rerr)**2)
                poog2=poog2.convert().coef
                if np.any( (poog2>10) | (poog2<-10)):
                    poog2 = np.polynomial.chebyshev.Chebyshev.fit((rwave),(rnorm),deg=order)
                    poog2=poog2.convert().coef
                #poog2[poog2>10]=10
                #poog2[poog2<-10]=-10
                poly0+=[0 for x in poog2]
                # change this back nov 20
            '''
            poly0=[0]*((order+1)*len(lam))
            print("contfit",poly0)
            p0=[*s_lab[s],*poly0,0,16000/22500 ]#np.full_like(s_lab[s],0.0)
            p0=np.array(p0)
            #preopt,coo,coo2 = fit_one_rv(theNN,trwav,dic,norma,err,lam,p0) 
            #peropt=[*s_lab[s],*preopt]
            #print("pi",peropt)
            
            '''
            f,a=plt.subplots(figsize=(100,5))
            newd,newe = gen_predict(dic,p0,theNN,trwav,lam)
            for k in range(len(lam)):
                cmask=np.ceil(np.interp(lam[k],cwav,contmask)).astype(bool)
                a.plot(lam[k],norma[k],color='k')
                a.fill_between(lam[k],norma[k]-err[k],norma[k]+err[k],color='gray',alpha=0.5)

                a.plot(lam[k],newd[k],color="darkred")
                a.fill_between(lam[k],newd[k]-newe[k],newd[k]+newe[k],color="red",alpha=0.5)
                a.plot(lam[k][cmask],norma[k][cmask],marker='o',linewidth=0,markersize=2,alpha=0.1) 
            a.set_ylim(0,1.1)
            #a.set_xlim(15150,15300)
            a.set_title(f"sdss_id: {ids[s]}")
            #plt.savefig(f"occam_womps/womp_{ids[s]}.png")
            plt.savefig("womp.png")
            '''

            popt,chi_i,chi_f=fit_one_ls(theNN,trwav,dic,norma,err,lam,p0,mask)
            #write_to(dic["direct"],st,chi_i,chi_f,p0,popt)       
            #print(p0)

            p0u=unnormalize_labels(p0[:-2-len(lam)*(1+order)],theNN)
            poptu=unnormalize_labels(popt[:-2-len(lam)*(1+order)],theNN)
            print("opt",popt)
            print("delta p",[f"{x:.3f}" for x in poptu-p0u])

            '''      
            f,a=plt.subplots(figsize=(100,5))
            newd,newe = gen_predict(dic,popt,theNN,trwav,lam)
            for k in range(len(lam)):
                a.plot(lam[k],norma[k],color='k')
                a.fill_between(lam[k],norma[k]-err[k],norma[k]+err[k],color='gray',alpha=0.5)
                

                a.plot(lam[k],newd[k],color="darkred")
                a.fill_between(lam[k],newd[k]-newe[k],newd[k]+newe[k],color="red",alpha=0.5)
                
                #a.fill_between(trwav,-999*mask,999*mask,color="blue",alpha=0.5)
                a.plot(lam[k],(newd[k]-norma[k]),color='green')
            a.set_ylim(0,1.1)
            #a.set_xlim(15150,15300)
            plt.savefig("womp.png")
            #plt.show()
            '''

            ci_s.append(chi_i)
            cf_s.append(chi_f)
            p0_s.append(p0u)
            pf_s.append(poptu)
            ii_s.append(ids[s])
        try:
            pass
        except:
            print("failure")
    write_to_all_a("",str(s4),"_"+os.path.basename(s0).replace(".txt","")+os.path.basename(dic["tag"])+"_"+os.path.basename(s15).replace(".txt",""),ii_s,ci_s,cf_s,p0_s,pf_s,fgz)
