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
import emcee,corner

from fit_one import *

def load_spectra_i(filt,nn,lims,totaltasks,taskid):
    # read in list of spectra locations
    fils=np.loadtxt(filt,dtype=str)
    # split up, if parallel process
    fils = fils[taskid::totaltasks]    

    waves=[]
    fluxs=[]
    errrs=[]

    #using summary file to make this quicker
    #t=Table.read("astraAllStarASPCAP-0.6.0.fits.gz",hdu=2)
    t=Table.read("trunc_raw_astraAllStarASPCAP-0.6.0.fits.gz",hdu=1)




    oo=[]
    # looping through spectra in the chunk
    for fil in fils:
        # load in spectra
        wave, flux, errs, model, mask, h = load_mwmStar(fil )
        # append sdss_id, which will be used to identify the star
        oo.append(int(os.path.basename(fil[:-5]).split("-")[-1]) )
        # regularizing errors
        errs[np.isinf(errs)|np.isnan(errs)]=999999.
        flux[np.isinf(flux)|np.isnan(flux)]=0.
        #errs[flux>1.01]=999999.
        errs[mask!=0]=999999.
        waves.append(wave)
        fluxs.append(flux)
        errrs.append(errs)


    # the labels of the neural network, hardcoded
    plist=['teff','logg','v_micro','fe_h','c_h','n_h','mg_h','al_h','mn_h','na_h','ca_h','ni_h','ti_h','si_h'] 
    plist=[f"raw_{x}" for x in plist]
    params = np.array([list(t[t['sdss_id'] == sid][plist][0].as_void()) for sid in oo])    
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
    return waves, s_labelz, fluxs, errrs, oo,flagl 

def write_to_all_a(direct,st,ii,chi_i,chi_f,true_lab,fit_lab,flagl):
    assert (len(chi_i)==len(chi_f)) and (len(true_lab)==len(fit_lab))
    pafz=os.path.join(direct,f"rparamfit{st}.txt")
    print(pafz)
    with open(pafz,'a') as mf:
        for q in range(len(chi_i)):
            true_string=" ".join([f"{la:.4f}" for la in true_lab[q]])
            fit_string=" ".join([f"{la:.4f}" for la in fit_lab[q]])
            flag_string=" ".join([f"{int(la)}" for la in flagl[q]])
            mf.write(f"{ii[q]} {chi_i[q]:.1f} {true_string}  {chi_f[q]:.1f} {fit_string} {flag_string}\n")

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
    for s in range(len(spec)):
        print(s)
        if True:
            #lam,norma,err=resample(wav,spec[s])
            lam=[wav[s]]
            norma=[spec[s]]
            err=[errs[s]]
            #a.plot(wav[s],errs[s],color='r')
            #plt.show()

            p0=[*s_lab[s],0,16000/22500 ]#np.full_like(s_lab[s],0.0)
            p0=np.array(p0)
            #preopt,coo,coo2 = fit_one_rv(theNN,trwav,dic,norma,err,lam,p0) 
            #peropt=[*s_lab[s],*preopt]
            #print("pi",peropt)

            popt,chi_i,chi_f=fit_one_ls(theNN,trwav,dic,norma,err,lam,p0,mask)
            #write_to(dic["direct"],st,chi_i,chi_f,p0,popt)       
            #print(p0)

            p0u=unnormalize_labels(p0[:-2],theNN)
            poptu=unnormalize_labels(popt[:-2],theNN)
            print(popt)
            print("delta p",[f"{x:.3f}" for x in poptu-p0u])

            '''       
            f,a=plt.subplots()
            for k in range(len(lam)):
                a.plot(lam[k],norma[k],color='k')
                a.fill_between(lam[k],norma[k]-err[k],norma[k]+err[k],color='gray',alpha=0.5)
                newd,newe = gen_predict(dic,popt,theNN,trwav,lam[k])

                a.plot(lam[k],newd,color="darkred")
                a.fill_between(lam[k],newd-newe,newd+newe,color="red",alpha=0.5)
                popt[8]+=0.5
                newd,newe = gen_predict(dic,popt,theNN,trwav,lam[k])

                a.plot(lam[k],newd,color="navy")
                
                #a.fill_between(trwav,-999*mask,999*mask,color="blue",alpha=0.5)
                a.plot(lam[k],(newd-norma[k]),color='green')
            a.set_ylim(-.5,1.1)
            a.set_xlim(15150,15300)

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
    write_to_all_a("","_"+os.path.basename(s0).replace(".txt","")+os.path.basename(dic["tag"])+"_"+os.path.basename(s15).replace(".txt",""),ii_s,ci_s,cf_s,p0_s,pf_s,fgz)
