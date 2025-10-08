#import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
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
print("loadt")

def read_in_neural_network(parz):
    '''
    read in the weights and biases parameterizing a particular neural network.
    You can read in existing networks from the neural_nets/ directory, or you
    can train your own networks and edit this function to read them in.
    '''

    tmp = np.load(parz)
    wlist=[]
    blist=[]
    i=0
    loop=True
    while loop:
        try:
            w_a=tmp[f"w_array_{i}"]
            b_a=tmp[f"b_array_{i}"]
            wlist.append(w_a)
            blist.append(b_a)
            i+=1
        except:
            loop=False
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    NN_coeffs = (*wlist, *blist, x_min, x_max)
    tmp.close()
    return NN_coeffs

def parse_input_file(paff,index=0):
    '''
    This just parses the input file, using the fixed format of the file, and returns the arguments as a dictionary
    '''
    ####training_labels, training_spectra, validation_labels, validation_spectra,wavelength,\
    #num_neurons=300,num_steps=1e4,learning_rate=1e-4,batch_size=512,tag="",direct="",save=True    
    split=0.3

    dicto={}
    numb=-1
    with open(paff,'r') as op:
        for line in op:
            text=line.strip()
            if text=="" or len(text)==0:
                continue
            elif text[0]=="#" or text[0]=="%":
                continue
            else:
                valu=""

                equals=text.find("=")
                if equals==-1:
                    # non-assignment line
                    if numb<index:
                        numb+=1
                    else:
                        break
                else:
                    if index==numb:
                        keyw=text[:equals].strip()
                        valu=text[equals+1:].strip()
                        # correctly type inputs
                        if valu=="True" or valu=="true":
                            valu=True
                        elif valu=="false" or valu=="False":
                            valu=False
                        elif valu.isnumeric():
                            valu=int(valu)
                        elif valu.replace("e","").replace("E","").isnumeric():
                            valu=int(float(valu))
                        elif valu.replace(".","").replace("-","").replace("e","").replace("E","").isnumeric():
                            valu=float(valu)
                        dicto.update({keyw:valu})
    try:
        inpu=dicto.pop("input")
    except:
        inpu=None
        # we don't need an input to simply call the trained thing
        #raise NameError("No input entered! You need to specify a dataset (.fits) to train on.")
    try:
        split=dicto.pop("split")
    except:
        pass
    try:
        direct=dicto["direct"]
        '''
        if not os.path.exists(direct):
            print("Making directory: ",direct)
            try:
                os.makedirs(direct)
            except:
                pass
        else:
            print(direct,": doesnt need to be made")
        '''
        assert os.path.exists(direct)
    except:
        raise NameError("This file needs to go somewhere... please specify an output directory.")
    try:
        tag=dicto["tag"]
    except:
        tag='_'+os.path.basename(fil).replace('.fits','')
        dicto["tag"]=tag
    beg=15000.
    end=17000.1
    try:
        beg=dicto.pop("beg")
    except:
        pass
    try:
        end=dicto.pop("end")
    except:
        pass
    return dicto,inpu,split,(beg,end)

def load_mwmStar(fname):
    with fits.open(fname) as hdul:
        h = dict(hdul[0].header)
        t = Table(hdul[3].data)
        if len(t) == 0:
            t = Table(hdul[4].data)
            #print(hdul[4].header)
        if len(t) > 1:
            print(f"{fname} has {len(t)} spectra, taking 0th")
        r = t[0]
        #print(r.colnames)
        wave = r["wavelength"]
        flux = r["flux"]/r["continuum"]
        errs = 1/np.sqrt(r["ivar"])/r["continuum"]
        mask = r["pixel_flags"] != 0
        model = r["nmf_rectified_model_flux"]#*r["continuum"]
        #print(r["nmf_flags"], not not r["nmf_flags"])
        if r["nmf_flags"]:
            print(f"{fname} failed continuum, {r['nmf_rchi2']}")
    return wave, flux, errs, model, mask, h

def normalize_labels(unscaled_labels, NN_coeffs):
    
    x_max=NN_coeffs[-1]
    x_min=NN_coeffs[-2]   
    with np.errstate(invalid='ignore'):
        x = (unscaled_labels-x_min)/(x_max-x_min) -0.5
    return np.where(np.isnan(x),0,x)

def unnormalize_labels(scaled_labels, NN_coeffs):
    
    x_max=NN_coeffs[-1]
    x_min=NN_coeffs[-2]
    with np.errstate(invalid='ignore'):
        x = (scaled_labels+0.5)*(x_max-x_min)+x_min
    return np.where(np.isnan(x),0,x)
    

def get_spectrum_from_neural_net(dic,scaled_labels, NN_coeffs):
    '''
    Predict the rest-frame spectrum (normalized) of a single star.
    We input the scaled stellar labels (not in the original unit).
    Each label ranges from -0.5 to 0.5
    '''
    
    def leaky_relu(z):
        '''
        This is the activation function used by default in all our neural networks.
        '''
        return z*(z > 0) + 0.01*z*(z < 0)
    
    def sigmoid(z):
        '''
        This is the activation function used by default in all our neural networks.
        '''
        return 1/(1+np.exp(-z))
    
    def hardtanh(z):
        return z+(-z)*(z<0)+(1-z)*(z>1)

    # assuming your NN has two hidden layers.
    #w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs
    w_and_b = NN_coeffs[:-2]
    if dic==None:
        hln=int((len(w_and_b)-2)/2)
        fisi=True
    else:
        fisi=dic['finalsigmoid']
        hln=dic['hiddenlayers']
        assert len(w_and_b)==2*hln+2 

    w_arrays = w_and_b[:hln+1]
    b_arrays = w_and_b[hln+1:]    
    
    '''
    inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
    outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
    spectrum = np.einsum('ij,j->i', w_array_2, sigmoid(outside)) + b_array_2
    return sigmoid(spectrum)
    '''
    #inside = np.einsum('ij,...j->...i', w_array_0, scaled_labels) + b_array_0
    #outside = np.einsum('ij,...j->...i', w_array_1, leaky_relu(inside)) + b_array_1
    #spectrum = np.einsum('ij,...j->...i', w_array_2, leaky_relu(outside)) + b_array_2
    #return sigmoid(spectrum)
    
    inside = np.einsum('ij,...j->...i', w_arrays[0], scaled_labels) + b_arrays[0]
    #print(w_arrays[0].shape,b_arrays[0].shape)
    for i in range(hln):
        #print(w_arrays[1+i].shape,b_arrays[1+i].shape)
        inside = np.einsum('ij,...j->...i', w_arrays[1+i], leaky_relu(inside) ) + b_arrays[1+i]
    
    if fisi:
        return sigmoid(inside)
    else:
        return inside

def load_sim_spectra(filt,nn,lims,noox=2000):
    fil='/project2/kicp/pthibodeaux/genspec/'
    fil=''
    
    with fits.open(fil+filt) as fi:
        wavelengths=fi[3].data
    
        labelz=fi[2].data[:noox]
        spectz=fi[1].data[:noox]
    
    print(labelz.T,lims)
    s_labelz=normalize_labels(labelz,nn)

    whas=np.where((wavelengths >= lims[0]) & (wavelengths < lims[1]))
    wavelengths=(wavelengths[whas])
    spectz=np.squeeze(spectz[...,whas])
    print(wavelengths.shape,spectz.shape)

    return wavelengths, s_labelz, spectz


def load_spectra(filt,nn,lims,noox=2000):
    #fils=os.listdir(filt)
    fils=np.loadtxt(filt,dtype=str)

    waves=[]
    fluxs=[]
    errrs=[]
    t=Table.read("../PISN_search/trunc_els_astraAllStarASPCAP-0.6.0.fits.gz",hdu=1)

    print(t.colnames)
    #print(t['sdss_id'])
    x=t["sdss_id"]
    oo=[]
    for fil in fils:
        wave, flux, errs, model, mask, h = load_mwmStar(fil )
        oo.append(int(os.path.basename(fil[:-5]).split("-")[-1]) )
        errs[np.isinf(errs)|np.isnan(errs)]=999999.
        flux[np.isinf(flux)|np.isnan(flux)]=0.
        errs[flux>1.01]=999999.
        errs[mask!=0]=999999.
        waves.append(wave)
        fluxs.append(flux)
        errrs.append(errs)
    
    tii = t[np.isin(x,oo)]
    teffs = tii['teff'].data
    params=[]
    for lab in ['teff','logg','v_micro','fe_h']:
         params.append(tii[lab].data)
    for lab in ['c_h','n_h','mg_h']:
         params.append(tii[lab].data-tii['fe_h'].data)
    #print(tii['v_rad'],tii['gaia_v_rad'])
    params=np.array(params).T
    print(params)    

    s_labelz=normalize_labels(params,nn)
    s_labelz[s_labelz<-0.5]=-0.5
    s_labelz[s_labelz>0.5]=0.5
    
    return waves, s_labelz, fluxs, errrs  

def load_NN(s2):
    fil='/project2/kicp/pthibodeaux/genspec/'
    fil=''

    NN_red = read_in_neural_network(fil+s2)
    return NN_red


def convo(spectrum,amount,waa):
    dr=waa[1]-waa[0]
    win=norm.pdf( (np.arange(21)-10.)*dr, scale=amount/2.35)
    win = win/np.sum(win)
    return convolve(spectrum,win,mode='same')

def resample(wavelengths, spect,eee=0.0001):
    lambda_blue, lambda_green, lambda_red = wavelength_solution(dr=16)
    lambd = [lambda_blue, lambda_green, lambda_red]
    
    amount=16000/22500
    norms0=convo(spect,amount,wavelengths)
    sampler=interp1d(wavelengths,norms0)

    normspec = [sampler(l)+np.random.normal(scale=eee,size=len(l)) for l in lambd]
    errsspec = [np.full_like(n,eee) for n in normspec]

    return lambd,normspec,errsspec

def gen_predict(dic,nlabels,NN_red,wavel,wave,mask=None,maskw=None):
    # wavel is the wavelengths of the neural network
    # wave is the wavelengths of the data

    # at given nlabels, generate spectrum from emulator
    spec_predict=get_spectrum_from_neural_net(dic,nlabels[:-2], NN_red)
    # convolve with broadening parameter
    spec_predict=convo(spec_predict,nlabels[-1],wavel)
    # doppler shift using rv parameter, then resample onto original wavelength grid
    vc=nlabels[-2]/3e5
    wavel2=wavel*np.sqrt((1+vc)/(1-vc))
    spec_predict=np.interp(wavel,wavel2,spec_predict)
    predict=interp1d(wavel2,spec_predict)
    # if we were working in a really sparse sampling the order and implementation of these would be really important
    # but generally, our NNE is trained at a higher resolution than the data
  
    # output in shape of data (for direct fitting) 
    norm_predict = np.zeros_like(wave)
    errs_predict = np.ones_like(wave)*99999.
     
    # find region of output that is represented by data, error will be set to 99999. elsewhere
    theone = (wave>wavel2[0])&(wave<wavel2[-1])
    allow_wave = wave[theone]

    norm_pred=predict(allow_wave)
    # errors are set to arbitrary sensitivity of our modeling. data sensitivity in APOGEE SNR~100 so 
    # final error during fitting should be dominated by data, not model, error
    errs_pred=np.ones_like(norm_pred)*0.001

    norm_predict[theone] = norm_pred
    errs_predict[theone] = errs_pred
    # if masking, block out masked region using larger errors
    if not mask is None:
        if maskw is None:
            mask2=np.interp(wave,wavel,mask)>0
        else:
            mask2=np.interp(wave,maskw,mask)>0
        errs_predict[mask2] = 99999.
    return norm_predict,errs_predict

def fit_func(nlabels,NN_red,normspec,errsspec,lambd,amount,dic,wavel,mask=None,maskw=None):
    wave_predict=np.concatenate(lambd)
    norm_predict,errs_predict=gen_predict(dic,nlabels,NN_red,wavel,wave_predict,mask,maskw)

    errs = np.sqrt( errs_predict**2 + np.concatenate(errsspec)**2)
    errs[(~np.isfinite(errs)) | (errs > 300) | (errs<0)] =999.
    resid = (np.concatenate(normspec)-norm_predict)/errs
    return resid
 
def fit_one_ls(NN_red,wave,dic,normspec,errsspec,lambd,p0,mask=None,maskw=None):
    npar=(NN_red[0].shape[-1])+2
    bound0=np.full(npar,-0.5)
    bound1=np.full(npar,0.5)
    bound0[-2]=-200
    bound0[-1]=0
    bound1[-2]=200
    bound1[-1]=5

    amount=16000/22500

    tol = 1e-7#5e-5

    ar=(NN_red,normspec,errsspec,lambd,amount,dic,wave,mask,maskw)
    res = least_squares(fit_func, p0,ftol=tol, xtol=tol, method='trf',bounds=(bound0,bound1),args=ar)
    popt = res.x

    chi_i= np.sum( fit_func(p0,*ar)**2)
    chi_f=np.sum( fit_func(popt,*ar)**2)
    return popt,chi_i,chi_f

def fit_func_rv(nlabels,NN_red,normspec,errsspec,lambd,p0,dic,wavel):
    wave_predict=np.concatenate(lambd)
    act_nlabels=np.copy(p0)
    act_nlabels[-2:]=nlabels
    #print("label eval",nlabels,act_nlabels)
    norm_predict,errs_predict=gen_predict(dic,act_nlabels,NN_red,wavel,wave_predict)
    #print(np.any(np.isinf(norm_predict)),np.any(np.isinf(errs_predict)))

    errs = np.sqrt( errs_predict**2 + np.concatenate(errsspec)**2)
    errs[(~np.isfinite(errs)) | (errs > 300) | (errs<0)] =999.
    resid = (np.concatenate(normspec)-norm_predict)/errs
    return resid

def fit_one_rv(NN_red,wave,dic,normspec,errsspec,lambd,p0):
    npar=(NN_red[0].shape[-1])+2
    bound0=np.full(2,-0)
    bound1=np.full(2,0)
    bound0[-2]=-200
    bound0[-1]=0
    bound1[-2]=200
    bound1[-1]=5

    amount=16000/22500

    tol = 1e-6#5e-5

    ar=(NN_red,normspec,errsspec,lambd,p0,dic,wave)
    res = least_squares(fit_func_rv, p0[-2:],ftol=tol, xtol=tol, method='trf',bounds=(bound0,bound1),args=ar)
    popt = res.x

    chi_i= np.sum( fit_func_rv(p0[-2:],*ar)**2)
    chi_f=np.sum( fit_func_rv(popt,*ar)**2)
    return popt,chi_i,chi_f


def write_to(direct,st,chi_i,chi_f,true_lab,fit_lab):
    '''
    with open(f"paramfit_input_{st}.txt",'a') as mf:
        true_string=" ".join([f"{la:.2f}" for la in true_lab])
        mf.write(f"{tag} {chi_i:.1f} {true_string}\n")
    with open(f"paramfit_output_{st}.txt",'a') as mf:
        fit_string=" ".join([f"{la:.2f}" for la in fit_lab])
        mf.write(f"{tag} {chi_f:.1f} {fit_string}\n")
    '''
    with open(os.path.join(direct,f"paramfit_output{st}.txt"),'a') as mf:
        true_string=" ".join([f"{la:.4f}" for la in true_lab])
        fit_string=" ".join([f"{la:.4f}" for la in fit_lab])
        mf.write(f"{chi_i:.1f} {true_string}  {chi_f:.1f} {fit_string}\n")

def write_to_all(direct,st,chi_i,chi_f,true_lab,fit_lab):
    assert (len(chi_i)==len(chi_f)) and (len(true_lab)==len(fit_lab))
    pafz=os.path.join(direct,f"paramfit_output{st}.txt")
    print(pafz)
    with open(pafz,'w') as mf:
        for q in range(len(chi_i)):
            true_string=" ".join([f"{la:.4f}" for la in true_lab[q]])
            fit_string=" ".join([f"{la:.4f}" for la in fit_lab[q]])
            mf.write(f"{chi_i[q]:.1f} {true_string}  {chi_f[q]:.1f} {fit_string}\n")

if __name__=="__main__":
    s0=sys.argv[1] # spectra data
    s1=sys.argv[2]
    s2=int(sys.argv[3])
    dic,inpu,x,lims=(parse_input_file(s1,s2))
    print(dic,inpu)

    nnpath=os.path.join("../smallscaleNNtest",dic["direct"],"NN_normalized_spectra"+dic["tag"]+".npz") 

    theNN=load_NN(nnpath)
    wav,s_lab,spec,errs=load_spectra(s0,theNN,lims)
       
    trwav = np.linspace(15000,17000,20001)
    s=1e4/trwav
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
    trwav =trwav*n
  
    #raise NameError()
    #s_lab=normalize_labels(lab,theNN)   

    ci_s=[]
    cf_s=[]
    p0_s=[]
    pf_s=[] 
    for s in range(len(spec)):
        print(s)
        if True:
            #lam,norma,err=resample(wav,spec[s])
            lam=[wav[s]]
            print("waves",wav[s].shape,wav[s])
            norma=[spec[s]]
            err=[errs[s]]
            #a.plot(wav[s],errs[s],color='r')
            #plt.show()

            p0=[*s_lab[s],0,16000/22500 ]#np.full_like(s_lab[s],0.0)

         
            p0=np.array(p0)
            #preopt,coo,coo2 = fit_one_rv(theNN,trwav,dic,norma,err,lam,p0) 
            #peropt=[*s_lab[s],*preopt]
            #print("pi",peropt)
            peropt=p0

            popt,chi_i,chi_f=fit_one_ls(theNN,trwav,dic,norma,err,lam,peropt)
            #write_to(dic["direct"],st,chi_i,chi_f,p0,popt)       
            #print(p0)
            #print("p0",p0)
            #print("pf",popt)

            print(p0,popt)
            p0u=unnormalize_labels(p0[:-2],theNN)
            poptu=unnormalize_labels(popt[:-2],theNN)
     
            #print("p0 unnorm",p0u)
            #print("pf unnorm",poptu)
            print("delta p",[f"{x:.3f}" for x in poptu-p0u])
            
            f,a=plt.subplots()
            for k in range(len(lam)):
                a.plot(lam[k],norma[k],color='k')
                a.fill_between(lam[k],norma[k]-err[k],norma[k]+err[k],color='gray',alpha=0.5)
                newd,newe = gen_predict(dic,popt,theNN,trwav,lam[k])

                a.plot(lam[k],newd,color="darkred")
                a.fill_between(lam[k],newd-newe,newd+newe,color="red",alpha=0.5)
                a.plot(lam[k],(newd-norma[k]),color='green')
            a.set_ylim(-.5,1.1)
            plt.show()
            
            ci_s.append(chi_i)
            cf_s.append(chi_f)
            p0_s.append(p0)
            pf_s.append(popt)
        try:
            pass
        except:
            print("failure")
    #write_to_all("","_"+s0.replace(".txt",""),ci_s,cf_s,p0_s,pf_s)
