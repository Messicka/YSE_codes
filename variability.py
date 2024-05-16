#!/usr/bin/env python3

from qso_fit import *
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting
import os

def dict_check(dic):
	check1 = dic['class']=='QSO'
	check2 = dic['signif_vary'] > 2
	check2 *= dic['signif_qso'] > 2
	check2 *= dic['signif_qso'] > dic['signif_not_qso']
	return int(check1 or check2)

def clean_flux(time, flux, flux_err, sigma=5):
    clip = sigma_clip(flux, sigma)
    mask = ~clip.mask * (flux > 0)
    return time[mask], flux[mask], flux_err[mask] 

def flux_to_mag(flux, flux_err):
    mag = 23.9 - 2.5 * np.log10(flux)
    mag_err = 2.5 * flux_err / (np.log(10) * flux)
    return mag, mag_err

def flux_check(time, flux, flux_err, color, n_min=10, point_test=False, noise_test=False):
    t, f, err = clean_flux(time, flux, flux_err)
    if point_test:
        where = np.abs(mag-mean_mag).argmax()
        t, f, err = np.delete(t,where), np.delete(f,where), np.delete(err,where)
    if noise_test:
        where = mag_err.argmax()
        t, f, err = np.delete(t,where), np.delete(f,where), np.delete(err,where)
    n_obs = t.size
    if n_obs > n_min:
        mag, mag_err = flux_to_mag(f, err)
        qso_dict = qso_fit(t, mag, mag_err, color)
    else:
        qso_dict = dict.fromkeys(qso_out, np.nan)
        qso_dict['class'] = 'not_qso'
    return n_obs, qso_dict

n_boot = 100
bands = ['g','r','i','z','y']
cols = ['dGal_num','filt','mean_mag','mag_stddev','n_obs','n_obs_std','obs_flag','lc_flag','frac_var','frac_unc',
	'lvar','lvar_err','ltau','ltau_err','chi2/nu','chi2/nu_err','nu','nu_err',
	'chi2_qso/nu','chi2_qso/nu_err','chi2_qso/nu_NULL','chi2_qso/nu_NULL_err',
	'signif_qso','signif_qso_err','signif_not_qso','signif_not_qso_err',
	'signif_vary','signif_vary_err','mode_class','pct_QSO','point_test','noise_test']
qso_out = ['lvar','ltau','chi2/nu','nu','chi2_qso/nu','chi2_qso/nu_NULL','signif_qso','signif_not_qso','signif_vary','class']

target_file = "targets/YSE_targets.csv"
outfile = "var_results.csv"
if os.path.isfile(outfile): results = pd.read_csv(outfile, usecols=cols)
else: results = pd.DataFrame(columns=cols)

lc_dir = 'dGal'
file_list = os.listdir(lc_dir)
l = len(file_list)
j = results['dGal_num'].unique().size
for filename in file_list:
    obj = filename.split('_')[0]
    num = int(obj[-5:])
    if num in results['dGal_num'].values: continue
    infile = os.path.join(lc_dir, filename)
    try: df = pd.read_table(infile, sep='\s+').dropna()
    except:
        print(f"Empty dataset for {obj}")
        os.remove(infile)
        continue
    clean_df = df[df['dq']==0].sort_values('mjd')
    for color in bands:
        c_data = clean_df[clean_df['filt']==color]		#selecting data corresponding to one band
        if not c_data.size: continue
        gal_flag = c_data['gal_flag'].values[0]
        time, flux, flux_err = c_data.loc[:,['mjd','flux','flux_err']].T.values
        main_time, main_flux, main_flux_err = clean_flux(time, flux, flux_err, sigma=5)
        main_mag, main_err = flux_to_mag(main_flux, main_flux_err)
        mean_mag, std_mag = main_mag.mean(), main_mag.std()

        flux_var = main_flux.var()
        mean_err2 = (main_flux_err**2).mean()
        if flux_var >= mean_err2:
            mean2_flux = flux.mean()**2
            frac_var = np.sqrt((flux_var-mean_err2) / mean2_flux)
            NXS_err = np.sqrt((np.sqrt(2/main_flux.size) + 4*frac_var**2/main_flux.size) * mean_err2 / mean2_flux)
            var_unc = np.sqrt(frac_var**2 + NXS_err) - frac_var
            flag = 0
        else:
            frac_var = np.nan
            var_unc = np.nan
            flag = 1

        if main_time.size > 10:
            p_where = np.abs(main_mag-mean_mag).argmax()
            p_test = dict_check(qso_fit(np.delete(main_time,p_where), np.delete(main_mag,p_where), np.delete(main_err,p_where), color))
            n_where = main_err.argmax()
            if n_where == p_where: n_test = p_test
            else: n_test = dict_check(qso_fit(np.delete(main_time,n_where), np.delete(main_mag,n_where), np.delete(main_err,n_where), color))
        else: p_test = n_test = 0

        boot_flux = np.random.normal(flux, flux_err, (n_boot,flux.size))
        obs_list, boot_list, qso_list = [], [], []
        for f in boot_flux:
            boot_obs, boot_dict = flux_check(time, f, flux_err, color)
            obs_list.append(boot_obs)
            boot_list.append(boot_dict.values())
            qso_list.append(dict_check(boot_dict))

        boot_df = pd.DataFrame(boot_list, columns=boot_dict.keys())
        classes, counts =  np.unique(boot_df['class'], return_counts=True)
        row_list = []
        for mean, std in zip(boot_df.iloc[:,:-1].mean(), boot_df.iloc[:,:-1].std()):
            row_list.append(mean)
            row_list.append(std)
        row = pd.DataFrame([[num,color,mean_mag,std_mag,np.mean(obs_list),np.std(obs_list),gal_flag,flag,frac_var,var_unc,*row_list,classes[counts.argmax()],np.mean(qso_list),p_test,n_test]],columns=cols)
        results = pd.concat((results,row),axis=0,ignore_index=True)
    j += 1
    print(f"{j} out of {l} done",end='\r')
print(f"{l} out of {l} done")

results = results.sort_values('dGal_num').reset_index(drop=True)
full_size = results['dGal_num'].nunique()
clean = results.loc[results['n_obs']>10].copy()
clean_size = np.unique(clean['dGal_num']).size
target_file = "targets/YSE_targets.csv"
target_data = pd.read_csv(target_file)

fit = fitting.LinearLSQFitter()
clip_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=5, sigma=3.0)
line_model = models.Linear1D()
fit_df = pd.DataFrame(columns=['m','b','filt'])

for i, c in enumerate(bands):
	x = clean.loc[clean['filt']==c,'mean_mag']
	y = clean.loc[clean['filt']==c,'mag_stddev']
	fitted_line, mask = clip_fit(line_model,x,y)
	add_df = pd.DataFrame([[*fitted_line.parameters,c]],columns=['m','b','filt'])
	fit_df = pd.concat((fit_df,add_df),ignore_index=True)

fit_df.to_csv('var_coeff.csv', index=False)
math_df = results[['dGal_num','mean_mag','mag_stddev','filt']].merge(fit_df,how='left')
results['line_test'] = (math_df['mag_stddev'] > math_df['m'] * math_df['mean_mag'] + math_df['b']).astype(int)
results['QSO'] = results['point_test'] * results['noise_test'] * results['line_test'] * (results['pct_QSO']>=0.5)

df_id = pd.read_csv(target_file)
results.merge(df_id.iloc[:,:-1], left_on='dGal_num', right_index=True).to_csv(outfile, index=False)
