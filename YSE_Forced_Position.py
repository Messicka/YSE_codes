#!/usr/bin/env python
# D. Jones - 6/21/23
### Query IPP forced photometry for any given ra,dec

import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import argparse
import os
import astropy.table as at
import astropy.units as u
from astropy.io import fits
from astropy.visualization import PercentileInterval, AsinhStretch
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from photutils.aperture import SkyCircularAperture, ApertureStats
import time
from io import BytesIO
import re
from lxml import html
import tempfile
import shutil
import pylab as plt
from astropy.time import Time

def date_to_mjd(date):
    time = Time(date,scale='utc')
    return time.mjd

def mjd_to_date(mjd):
    time = Time(mjd,format='mjd',scale='utc')
    return time.isot

### hack to get photometric band names - avoids one more API query ###
phot_band_dict = {'https://ziggy.ucolick.org/yse/api/photometricbands/216/':'g',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/217/':'r',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/218/':'i',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/219/':'z',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/220/':'y',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/36/':'g',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/37/':'r',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/39/':'i',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/40/':'z',
                  'https://ziggy.ucolick.org/yse/api/photometricbands/41/':'y'}

### This is FITS Header boilerplate stuff from Mark ###
default_stamp_header = fits.Header()
default_stamp_header['XTENSION'] = 'BINTABLE'
default_stamp_header['BITPIX']  = 8
default_stamp_header['NAXIS']   = 2
default_stamp_header['NAXIS1']  = 476
default_stamp_header['NAXIS2']  = 9
default_stamp_header['PCOUNT']  = 0
default_stamp_header['GCOUNT']  = 1
default_stamp_header['TFIELDS'] = 24
default_stamp_header['TTYPE1']  = 'ROWNUM  '
default_stamp_header['TFORM1']  = 'J       '
default_stamp_header['TTYPE2']  = 'PROJECT '
default_stamp_header['TFORM2']  = '16A     '
default_stamp_header['TTYPE3']  = 'SURVEY_NAME'
default_stamp_header['TFORM3']  = '16A     '
default_stamp_header['TTYPE4']  = 'IPP_RELEASE'
default_stamp_header['TFORM4']  = '16A     '
default_stamp_header['TTYPE5']  = 'JOB_TYPE'
default_stamp_header['TFORM5']  = '16A     '
default_stamp_header['TTYPE6']  = 'OPTION_MASK'
default_stamp_header['TFORM6']  = 'J       '
default_stamp_header['TTYPE7']  = 'REQ_TYPE'
default_stamp_header['TFORM7']  = '16A     '
default_stamp_header['TTYPE8']  = 'IMG_TYPE'
default_stamp_header['TFORM8']  = '16A     '
default_stamp_header['TTYPE9']  = 'ID      '
default_stamp_header['TFORM9']  = '16A     '
default_stamp_header['TTYPE10'] = 'TESS_ID '
default_stamp_header['TFORM10'] = '64A     '
default_stamp_header['TTYPE11'] = 'COMPONENT'
default_stamp_header['TFORM11'] = '64A     '
default_stamp_header['TTYPE12'] = 'COORD_MASK'
default_stamp_header['TFORM12'] = 'J       '
default_stamp_header['TTYPE13'] = 'CENTER_X'
default_stamp_header['TFORM13'] = 'D       '
default_stamp_header['TTYPE14'] = 'CENTER_Y'
default_stamp_header['TFORM14'] = 'D       '
default_stamp_header['TTYPE15'] = 'WIDTH   '
default_stamp_header['TFORM15'] = 'D       '
default_stamp_header['TTYPE16'] = 'HEIGHT  '
default_stamp_header['TFORM16'] = 'D       '
default_stamp_header['TTYPE17'] = 'DATA_GROUP'
default_stamp_header['TFORM17'] = '64A     '
default_stamp_header['TTYPE18'] = 'REQFILT '
default_stamp_header['TFORM18'] = '16A     '
default_stamp_header['TTYPE19'] = 'MJD_MIN '
default_stamp_header['TFORM19'] = 'D       '
default_stamp_header['TTYPE20'] = 'MJD_MAX '
default_stamp_header['TFORM20'] = 'D       '
default_stamp_header['TTYPE21'] = 'RUN_TYPE'
default_stamp_header['TFORM21'] = '16A     '
default_stamp_header['TTYPE22'] = 'FWHM_MIN'
default_stamp_header['TFORM22'] = 'D       '
default_stamp_header['TTYPE23'] = 'FWHM_MAX'
default_stamp_header['TFORM23'] = 'D       '
default_stamp_header['TTYPE24'] = 'COMMENT '
default_stamp_header['TFORM24'] = '64A     '
default_stamp_header['EXTNAME'] = 'PS1_PS_REQUEST'
default_stamp_header['REQ_NAME'] = 'yse.meh_stamp_testid200410'
default_stamp_header['EXTVER']  = '2       '
default_stamp_header['ACTION']  = 'PROCESS '
default_stamp_header['EMAIL']   = 'yse@qub.ac.uk'

default_forcedphot_header = fits.Header()
default_forcedphot_header['XTENSION'] = 'BINTABLE'
default_forcedphot_header['BITPIX']   = 8
default_forcedphot_header['NAXIS']    = 2
default_forcedphot_header['NAXIS1']   = 84
default_forcedphot_header['NAXIS2']   = 8
default_forcedphot_header['PCOUNT']   = 0
default_forcedphot_header['GCOUNT']   = 1
default_forcedphot_header['TFIELDS']  = 9
default_forcedphot_header['TTYPE1']   = 'ROWNUM  '
default_forcedphot_header['TFORM1']   = '20A     '
default_forcedphot_header['TTYPE2']   = 'RA1_DEG '
default_forcedphot_header['TFORM2']   = 'D       '
default_forcedphot_header['TTYPE3']   = 'DEC1_DEG'
default_forcedphot_header['TFORM3']   = 'D       '
default_forcedphot_header['TTYPE4']   = 'RA2_DEG '
default_forcedphot_header['TFORM4']   = 'D       '
default_forcedphot_header['TTYPE5']   = 'DEC2_DEG'
default_forcedphot_header['TFORM5']   = 'D       '
default_forcedphot_header['TTYPE6']   = 'FILTER  '
default_forcedphot_header['TFORM6']   = '20A     '
default_forcedphot_header['TTYPE7']   = 'MJD-OBS '
default_forcedphot_header['TFORM7']   = 'D       '
default_forcedphot_header['TTYPE8']   = 'FPA_ID  '
default_forcedphot_header['TFORM8']   = 'J       '
default_forcedphot_header['TTYPE9']   = 'COMPONENT'
default_forcedphot_header['TFORM9']   = '64A     '
default_forcedphot_header['EXTNAME']  = 'MOPS_DETECTABILITY_QUERY'
default_forcedphot_header['QUERY_ID'] = 'yse.meh_det_test200410'
default_forcedphot_header['EXTVER']   = '2       '
default_forcedphot_header['OBSCODE']  = '566     '
default_forcedphot_header['STAGE']    = 'WSdiff  '
default_forcedphot_header['EMAIL']    = 'yse@qub.ac.uk'

def getskycell(ra,dec):

	session = requests.Session()
	session.auth = ('ps1sc','skysurveys')
	skycellurl = 'http://pstamp.ipp.ifa.hawaii.edu/findskycell.php'
	
	# First login. Returns session cookie in response header. Even though status_code=401, it is ok
	page = session.post(skycellurl)

	info = {'ra': (None, ra), 'dec': (None, dec)}
	page = session.post(skycellurl, data=info)

	skycell = page.text.split("<tr><td>RINGS.V3</td><td>skycell.")[-1].split('</td>')[0]
	xpos = page.text.split("<tr><td>RINGS.V3</td><td>skycell.")[-1].split('<td>')[1].split('</td>')[0]
	ypos = page.text.split("<tr><td>RINGS.V3</td><td>skycell.")[-1].split('<td>')[2].split('</td>')[0]
	
	return skycell,xpos,ypos


### just a regex for the cameras using exp name ### 
def get_camera(exp_name):
    if re.match('o[0-9][0-9][0-9][0-9]g[0-9][0-9][0-9][0-9]o',exp_name):
        return 'GPC1'
    elif re.match('o[0-9][0-9][0-9][0-9]h[0-9][0-9][0-9][0-9]o',exp_name):
        return 'GPC2'
    elif re.match('o[0-9][0-9][0-9][0-9][0-9]g[0-9][0-9][0-9][0-9]o',exp_name):
        return 'GPC1'
    elif re.match('o[0-9][0-9][0-9][0-9][0-9]h[0-9][0-9][0-9][0-9]o',exp_name):
        return 'GPC2'
    else: raise RuntimeError('couldn\'t parse exp name')

### making PNG files ###
def fits_to_png(ff,outfile=None):
    plt.clf()
    ax = plt.axes()
    fim = ff[1].data
    # replace NaN values with zero for display
    fim[np.isnan(fim)] = 0.0
    # set contrast to something reasonable
    transform = AsinhStretch() + PercentileInterval(99.5)
    bfim = transform(fim)
    ax.imshow(bfim,cmap="gray",origin="lower")
    circle = plt.Circle((np.shape(fim)[0]/2-1, np.shape(fim)[1]/2-1), 15, color='r',fill=False)
    ax.add_artist(circle)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    if outfile is None: return plt
    else: plt.savefig(outfile,bbox_inches = 'tight',pad_inches = 0)

def fluxToMicroJansky(adu, exptime, zp):
    factor = 10**(-0.4*(zp-23.9))
    uJy = adu/exptime*factor
    return uJy

def parse_mdc(mdc_text):

    mdc = {}
    for line in mdc_text.split('\n'):
        if line.startswith('ROWNUM'):
            rownum = int(line.split()[-1])
            mdc[rownum] = {}
        elif line.startswith('ROW'): continue
        elif line.startswith('END'): continue
        elif not line: continue
        else:
            mdc[rownum][line.split()[0]] = line.split()[2]
            
    return mdc

def getRADecBox(ra,dec,size=None):
    RAboxsize = DECboxsize = size

    # get the maximum 1.0/cos(DEC) term: used for RA cut
    minDec = dec-0.5*DECboxsize
    if minDec<=-90.0:minDec=-89.9
    maxDec = dec+0.5*DECboxsize
    if maxDec>=90.0:maxDec=89.9

    invcosdec = max(1.0/np.cos(dec*np.pi/180.0),
                    1.0/np.cos(minDec  *np.pi/180.0),
                    1.0/np.cos(maxDec  *np.pi/180.0))

    ramin = ra-0.5*RAboxsize*invcosdec
    ramax = ra+0.5*RAboxsize*invcosdec
    decmin = dec-0.5*DECboxsize
    decmax = dec+0.5*DECboxsize

    if ra<0.0: ra+=360.0
    if ra>=360.0: ra-=360.0

    if ramin!=None:
        if (ra-ramin)<-180:
            ramin-=360.0
            ramax-=360.0
        elif (ra-ramin)>180:
            ramin+=360.0
            ramax+=360.0
    return(ramin,ramax,decmin,decmax)

class YSE_Forced_Pos:
    def __init__(self):
        pass

    def add_args(self,parser=None, usage=None, config=None):

        if parser == None:
            parser = argparse.ArgumentParser(
                usage=usage, conflict_handler="resolve")

        parser.add_argument('-n','--name', type=str, default=None,
                            help='give a name, just helps with bookkeeping')
        parser.add_argument('-r','--ra', type=float, default=None,
                            help='RA')
        parser.add_argument('-d','--dec', type=float, default=None,
                            help='Dec')
        parser.add_argument('-c','--coordlist', type=str, default=None,
                            help='coordinate list')
        parser.add_argument('-o','--outdir', type=str, default=None,
                            help='output directory for images and photometry files')
        parser.add_argument('--use_csv', default=False, action="store_true",
                            help="use CSV field for field info if set")
        parser.add_argument('--csv_filename', default='YSE_observations_full.csv', type=str,
                            help="CSV file name")
        parser.add_argument('--nofitsdownload', default=False, action="store_true",
                            help="don't download fits files if set")
        parser.add_argument('--checkfiles', default=False, action="store_true",
                            help="won't analyze objects for which files already exist")
 
        parser.add_argument('--ysepzurl',type=str,default='https://ziggy.ucolick.org/yse',
                            help='base URL for YSE-PZ, so you can reach the API and get field IDs')
        parser.add_argument('-u','--ysepzuser',type=str,default=None,
                            help='YSE-PZ username')
        parser.add_argument('-p','--ysepzpass',type=str,default=None,
                            help='YSE-PZ password')
        parser.add_argument('--ifauser',type=str,default='ps1sc',
                            help='IfA username')
        parser.add_argument('--ifapass',type=str,default='skysurveys',
                            help='IfA password')

        return parser

    def request_forcedphot(self,obs_data_dict,width=300,height=300,skycelldict=None,submitrequest=True):
        """
        A function that inputs a obs_data_dict for a table of transients, creates a stamp dictionary,
        and (optionally) sends the request
        """

        for i, k in enumerate(obs_data_dict.keys()):
            #print(obs_data_dict[k]['results'])
            #print(len(obs_data_dict[k]['results']))
            key_table = at.Table(obs_data_dict[k]['results'])['obs_mjd','photometric_band','diff_id','stack_id']
            if not self.options.use_csv:
                key_table['photometric_band'] = [phot_band_dict[band] for band in key_table['photometric_band']]
            key_table['transient_id'] = [k] * len(key_table)
            key_table['ra'] = [obs_data_dict[k]['ra']] * len(key_table)
            key_table['dec'] = [obs_data_dict[k]['dec']] * len(key_table)
            key_table['camera'] = [get_camera(obs['image_id']) for obs in obs_data_dict[k]['results']]
            obs_table = at.vstack([obs_table,key_table]) if i else key_table.copy()
        if not len(obs_table):
            return None,[],{},obs_table,0
        
        transient_unq,idx = np.unique(obs_table['transient_id'],return_index=True)
        if skycelldict is None:
            skycelldict = {}
            for snid,ra,dec in obs_table['transient_id','ra','dec'][idx]:
                skycelldict[snid] = 'skycell.'+getskycell(ra,dec)[0]
        
        request_names = []
        diff_count =  0
        diff_data = at.Table(names=('ROWNUM','PROJECT','RA1_DEG','DEC1_DEG','RA2_DEG','DEC2_DEG','FILTER','MJD-OBS','FPA_ID','COMPONENT_ID'),
                             dtype=('S20','S16','>f8','>f8','>f8','>f8','S20','>f8','>i4','S64'))
        for snid in transient_unq:
            id_match = (obs_table['transient_id'] == snid)
            for ra,dec,mjd,filt,camera,diff_id, in obs_table['ra','dec','obs_mjd','photometric_band','camera','diff_id'][id_match]:
                if diff_id is not None and diff_id != 'NULL':
                    diff_data.add_row((f'forcedphot_ysebot_{diff_count}',camera,ra,dec,ra,dec,filt,mjd,diff_id,skycelldict[snid]))
                    diff_count += 1

            if len(diff_data) > 0:
                hdr = default_forcedphot_header.copy()
                request_name = f'YSE-phot.{snid}.{diff_id}.{int(time.time())}'
                hdr['QUERY_ID'] = request_name
                hdr['EXTNAME'] = 'MOPS_DETECTABILITY_QUERY'
                hdr['EXTVER'] = '2'
                hdr['OBSCODE'] = '566'
                hdr['STAGE'] = 'WSdiff'
                ff = fits.BinTableHDU(diff_data, header=hdr)
                s = BytesIO()
                ff.writeto(s, overwrite=True)
                self.submit_to_ipp(s)
                request_names += [request_name]
            else:
                print(f'warning : no diff IDs for transient {snid_unq}')

        return request_names,skycelldict,obs_table,0         

    def request_templates(self,trans_data,width=300,height=300,skycelldict=None):
        
        data = at.Table(names=('ROWNUM', 'PROJECT', 'SURVEY_NAME', 'IPP_RELEASE', 'JOB_TYPE',
                               'OPTION_MASK', 'REQ_TYPE', 'IMG_TYPE', 'ID', 'TESS_ID',
                               'COMPONENT', 'COORD_MASK', 'CENTER_X', 'CENTER_Y', 'WIDTH',
                               'HEIGHT', 'DATA_GROUP', 'REQFILT', 'MJD_MIN', 'MJD_MAX',
                               'RUN_TYPE', 'FWHM_MIN', 'FWHM_MAX', 'COMMENT'),
                        dtype=('>i4','S16','S16','S16','S16','>i4','S16','S16','S16','S64',
                               'S64','>i4','>f8','>f8','>f8','>f8','S64','S16','>f8','>f8',
                               'S16','>f8','>f8','S64'))

        trans_unq, idx = np.unique(trans_data['transient_id'], return_index=True)
        if skycelldict is None:
            skycelldict = {}
            for trans_id,ra,dec in trans_data['stack_id','ra','dec'][idx]:
                skycelldict[trans_id] = 'skycell.'+getskycell(ra,dec)[0]

        count = 1
        for stack_id,ra,dec,camera,trans_id in trans_data:
            if stack_id is None: continue
            skycell_str = skycelldict[trans_id]
            data.add_row((count,camera,'null','null','stamp',2049,'byid','stack',stack_id,'RINGS.V3',
                          skycell_str,2,ra,dec,width,height,'null','null',0,0,'null',0,0,f'stack.for.{trans_id}'))
            count += 1

        hdr = default_stamp_header.copy()
        request_name = f'YSE-stamp.{int(time.time())}'
        hdr['REQ_NAME'] = request_name
        ff = fits.BinTableHDU(data, header=hdr)

        s = BytesIO()
        ff.writeto(s, overwrite=True)
        #if self.debug:
        #    ff.writeto('stampimg.fits',overwrite=True)

        self.submit_to_ipp(s)
        return request_name,skycelldict


    def get_phot(self,request_names,trans_data):
        trans_list = trans_data['transient_id']
        ra_list = trans_data['ra']
        dec_list = trans_data['dec'] 
        sct = SkyCoord(ra_list,dec_list,unit=u.deg)

        phot_dict = {}
        phot_link = 'http://datastore.ipp.ifa.hawaii.edu/pstampresults/'
        rn_len = len(request_names)
        for i, request_name in enumerate(request_names):
            print(f"Getting photometry: {i} out of {rn_len} done", end='\r')
            phot_results_link = f'{phot_link}/{request_name}/'
            phot_page = requests.get(url=phot_results_link)
            if phot_page.status_code != 200:
                raise RuntimeError('results page {phot_results_link} does not exist')

            tree = html.fromstring(phot_page.content)
            fitsfiles = tree.xpath('//a/text()')
            for f in fitsfiles:
                if 'detectability' in f:
                    phot_fits_link = f'{phot_link}/{request_name}/{f}'
                    fits_response = requests.get(url=phot_fits_link,stream=True)

                    # this is a pain but it seems necessary
                    tmpfits = tempfile.NamedTemporaryFile(delete=False)
                    shutil.copyfileobj(fits_response.raw, tmpfits)
                    tmpfits.close()
                    ff = fits.open(tmpfits.name)
                    os.remove(tmpfits.name)
                    for i in range(len(ff[1].data)):
                        mjd = ff[0].header['MJD-OBS']
                        exptime = ff[0].header['EXPTIME']
                        filt = ff[0].header['FPA.FILTER'].split('.')[0]
                        flux = ff[1].data['PSF_INST_FLUX'][i]
                        flux_err = ff[1].data['PSF_INST_FLUX_SIG'][i]
                        # http://svn.pan-starrs.ifa.hawaii.edu/trac/ipp/browser/trunk/psModules/src/objects/pmSourceMasks.h?order=name
                        # saturated, diff spike, ghost, off chip
                        #bad_flags = ['0x00001000','0x20000000','0x40000000','0x80000000']
                        # saturation, defect
                        #0x00000080, 0x00000800
                        if ff[1].data['PSF_QF'][i] < 0.9 or \
                           (ff[1].data['FLAGS'][i] & 0x00001000) or \
                           (ff[1].data['FLAGS'][i] & 0x20000000) or \
                           (ff[1].data['FLAGS'][i] & 0x40000000) or \
                           (ff[1].data['FLAGS'][i] & 0x80000000) or \
                           (ff[1].data['FLAGS'][i] & 0x00000080) or \
                           (ff[1].data['FLAGS'][i] & 0x00000800): dq = 1
                        else: dq = 0
                        try:
                            stack_id = ff[0].header['PPSUB.REFERENCE'].split('.')[-3]
                        #    warp_id = ff[0].header['PPSUB.INPUT'].split('.')[3]
                        except:
                            stack_id = None
                        #    warp_id = None
                        ra = ff[1].data['RA_PSF'][i]
                        dec = ff[1].data['DEC_PSF'][i]
                        sc = SkyCoord(ff[1].data['RA_PSF'][i],ff[1].data['DEC_PSF'][i],unit=u.deg)
                        sep = sc.separation(sct).arcsec
                        if np.min(sep) > 2:
                            ra, dec = ff[1].data['RA_PSF'][i], ff[1].data['DEC_PSF'][i]
                            raise RuntimeError(f'Couldn\'t find transient match for RA,Dec={ra:.7f},{dec:.7f}')
                        tn = trans_list[np.argmin(sep)]
                        if tn not in phot_dict.keys():
                            phot_dict[tn] = {'mjd':[],
                                             'filt':[],
                                             'diff_flux':[],
                                             'diff_err':[],
                                             'dq':[],
                                             'stack_id':[],
                                             #'warp_id':[],
                                             'diff_id':[],
                                             'ra':[],
                                             'dec':[],
                                             'exptime':[],
                                             'zpt':[],
                                             'camera':[]}

                        phot_dict[tn]['mjd'] += [mjd]
                        phot_dict[tn]['filt'] += [filt]
                        phot_dict[tn]['diff_flux'] += [flux]
                        phot_dict[tn]['diff_err'] += [flux_err]
                        phot_dict[tn]['dq'] += [dq]
                        phot_dict[tn]['stack_id'] += [stack_id]
                        #phot_dict[tn]['warp_id'] += [warp_id]
                        phot_dict[tn]['diff_id'] += [f.split('.')[2]]
                        phot_dict[tn]['ra'] += [ra]
                        phot_dict[tn]['dec'] += [dec]
                        phot_dict[tn]['exptime'] += [exptime]
                        phot_dict[tn]['zpt'] += [ff[0].header['FPA.ZP']]
                        phot_dict[tn]['camera'] += [ff[0].header['FPA.INSTRUMENT']]
        print(f"Getting photometry: {rn_len} out of {rn_len} done")
        return phot_dict

    def write_photometry(self,phot_dict,stack_dict):

        for t in phot_dict.keys():
            with open(f"{self.options.outdir}/{t}_phot.dat",'w') as fout:
                print('# mjd filt exp_time zpt diff_flux diff_err gal_flux gal_err flux flux_err dq',file=fout)
                phot_table = at.Table(phot_dict[t])
                stack_table = at.Table(stack_dict[t])
                data_table = at.join(phot_table,stack_table[['stack_image_id','gal_flux','gal_err']],
                                  join_type='left',keys_left='stack_id',keys_right='stack_image_id')
                data_table['dq'] = data_table['dq'].astype(float)
                data_table['flux'] = data_table['gal_flux'] + data_table['diff_flux']
                data_table['flux_err'] = np.sqrt(data_table['gal_err']**2 + data_table['diff_err']**2)
                for m,f,exp,zpt,df,dfe,gf,gfe,fl,fle,dq in data_table['mjd','filt','exptime','zpt','diff_flux','diff_err','gal_flux','gal_err','flux','flux_err','dq']:
                    print(f"{m:.3f} {f} {exp} {zpt:.4f} {df:.4f} {dfe:.4f} {gf:.4f} {gfe:.4f} {fl:.4f} {fle:.4f} {dq}",file=fout)
        
        return
    
    def get_stamps(self,request_name,coord_data):

        stamp_link = 'http://datastore.ipp.ifa.hawaii.edu/yse-pstamp-results/'
        stamp_fitsfile_link = stamp_link + f'{request_name}/'
        stamp_results_link = stamp_fitsfile_link + 'results.mdc'
        
        stamps_page = requests.get(url=stamp_results_link)
        if stamps_page.status_code != 200:
            raise RuntimeError(f'results page {stamp_results_link} does not exist')
        
        mdc_stamps = parse_mdc(stamps_page.text)

        #tree = html.fromstring(stamps_page.content)
        #fitsfiles = tree.xpath('//a/text()')
        
        image_dict = {}

        for k in mdc_stamps.keys():
            err = mdc_stamps[k]['ERROR_STR']
            #print(k, err)
            if 'SUCCESS' not in err:
                print(f'warning: part of job {request_name} failed! {k} has error {err}')
            #elif err in ['PSTAMP_NO_VALID_PIXELS','PSTAMP_NO_IMAGE_MATCH']:
            #    pass
            else:
                img_name,img_type,transient,mjd,img_id,img_filter,img_camera = \
                    mdc_stamps[k]['IMG_NAME'],mdc_stamps[k]['IMG_TYPE'],mdc_stamps[k]['COMMENT'].split('.')[-1],\
                    float(mdc_stamps[k]['MJD_OBS']),mdc_stamps[k]['ID'],mdc_stamps[k]['FILTER'].split('.')[0],\
                    mdc_stamps[k]['PROJECT']

                if transient not in image_dict.keys():
                    image_dict[transient] = {'stack_image_link':[],'stack_image_id':[],'stack_image_mjd':[],
                                             'stack_image_filter':[],'stack_image_camera':[],'ra':[],'dec':[]}
                id_match = (coord_data['stack_id']==img_id)
                if img_type == 'stack' and img_id not in image_dict[transient]['stack_image_id']:
                    image_dict[transient]['stack_image_link'] += [f'{stamp_fitsfile_link}/{img_name}']
                    image_dict[transient]['stack_image_id'] += [img_id]
                    image_dict[transient]['stack_image_mjd'] += [mjd]
                    image_dict[transient]['stack_image_filter'] += [img_filter]
                    image_dict[transient]['stack_image_camera'] += [img_camera]
                    image_dict[transient]['ra'] += [coord_data['ra'][id_match]]
                    image_dict[transient]['dec'] += [coord_data['dec'][id_match]]
                elif img_type not in ['stack','warp','diff']: 
                    raise RuntimeError(f'image type {img_type} not found')

        return image_dict
 
    def get_status(self,request_name,warning=False):
        
        status_link = 'http://pstamp.ipp.ifa.hawaii.edu/status.php'
        session = requests.Session()
        session.auth = (self.options.ifauser,self.options.ifapass)
        page = session.post(status_link)
        page = session.post(status_link)	#Why is this necessary? It is, but why?
        
        if page.status_code == 200:
            lines_out = []
            for line in page.text.split('<pre>')[-1].split('\n'):
                if line and '------------------' not in line and '/pre' not in line:
                    lines_out += [line[1:]]
            text = '\n'.join(lines_out)
            tbl = at.Table.read(text,format='ascii',delimiter='|',data_start=1,header_start=0)

            idx = tbl['name'] == request_name
            if not len(tbl[idx]):
                if warning: print(f'warning: could not find request named {request_name}')
                return False, False
            if tbl['Completion Time (UTC)'][idx]: done = True
            else: done = False

            jobs = float(tbl['Total Jobs'][idx])
            jobs_succ = float(tbl['Successful Jobs'][idx])
            success = jobs == jobs_succ
            if not success and warning: print(f'warning: {jobs-jobs_succ} of {jobs} jobs failed')
        else:
            print(f'Error occured with request {request_name}')
            done = success = False
        return done,success

    def get_gal_flux(self,img_dict):

        basedir = self.options.outdir
        session = requests.Session()
        session.auth = (self.options.ifauser,self.options.ifapass)

        for k in img_dict.keys():
            img_dict[k]['gal_flux'] = []
            img_dict[k]['gal_err'] = []
            if not self.options.nofitsdownload:
                img_dict[k]['stack_file'] = []
                img_dict[k]['stack_file'+'_png'] = []

            #session = requests.Session()
            #session.auth = (self.options.ifauser,self.options.ifapass)
            for i in range(len(img_dict[k]['stack_image_link'])):
                if img_dict[k]['stack_image_link'][i] is None:
                    img_dict[k]['gal_flux'] += [None]
                    img_dict[k]['gal_err'] += [None]
                    if not self.options.nofitsdownload:
                        img_dict[k]['stack_file'] += [None]
                        img_dict[k]['stack_file'+'_png'] += [None]
                    continue
                        
                outdir = f"{basedir}/{k}/{int(img_dict[k]['stack_image_mjd'][i])}"
                if not os.path.exists(outdir): os.makedirs(outdir)

                filename = img_dict[k]['stack_image_link'][i].split('/')[-1]
                outfile = f"{outdir}/{filename}"
                    
                fits_response = session.get(img_dict[k]['stack_image_link'][i],stream=True)
                with open(outfile,'wb') as fout:
                    shutil.copyfileobj(fits_response.raw, fout)

                ff = fits.open(outfile)

                pos = SkyCoord(img_dict[k]['ra'][i], img_dict[k]['dec'][i], unit=u.deg)
                ap = SkyCircularAperture(pos, r = 2.5*u.arcsec)
                ap_pix = ap.to_pixel(WCS(ff[1].header)) 
                apstats = ApertureStats(ff[1].data, ap_pix)
                factor =  10**-0.44 * ff[1].header['NINPUTS'] / ff[1].header['EXPTIME']
                img_dict[k]['gal_flux'] += [factor * apstats.sum[0]]
                img_dict[k]['gal_err'] += [factor * apstats.std[0]]

                if self.options.nofitsdownload:
                     shutil.rmtree(outdir)
                else:
                    fits_to_png(ff,outfile.replace('fits','png'))
                    img_dict[k]['stack_file'] += [outfile.replace('{basedir}/','')]
                    img_dict[k]['stack_file'+'_png'] += [outfile.replace('{basedir}/','').replace('.fits','.png')]

            updir = outdir.rsplit('/',1)[0]
            if not len(os.listdir(updir)):
                os.rmdir(updir)

        return img_dict

    def submit_to_ipp(self,filename_or_obj):

        session = requests.Session()
        session.auth = (self.options.ifauser,self.options.ifapass)
        stampurl = 'http://pstamp.ipp.ifa.hawaii.edu/upload.php'

        # First login. Returns session cookie in response header. Even though status_code=401, it is ok
        page = session.post(stampurl)

        if type(filename_or_obj) == str: files = {'filename':open(filename,'rb')}
        else: files = {'filename':filename_or_obj.getvalue()}
        page = session.post(stampurl, files=files)

    
    def main(self):

        if self.options.ra is not None and self.options.dec is not None and self.options.name is not None:
            if self.options.checkfiles and os.path.exists(f'{self.options.outdir}/{self.options.name}_phot.dat'): pass
            else: self.all_stages([self.options.name],[self.options.ra],[self.options.dec])
        elif self.options.coordlist is not None:
            name,ra,dec = np.loadtxt(self.options.coordlist,delimiter=',',dtype=str,unpack=True)
            ra,dec = ra.astype(float),dec.astype(float)
            keep = [not os.path.exists(f'{self.options.outdir}/{obj}_phot.dat') for obj in name] if self.options.checkfiles else np.ones_like(name, dtype=bool)
            if np.array(keep).sum(): self.all_stages(name[keep],ra[keep],dec[keep])

    def all_stages(self,namelist,ralist,declist):
        #### main function ####


        ### 1. query the YSE-PZ API for all obs matching ra/dec ###
        # returns diff_id, warp_id, filter, camera
        # this might be slow if the list is long!
        obs_data_dict = {}
        total_images = 0

        if self.options.use_csv:
            csvdata = at.Table.read(self.options.csv_filename,format='ascii.csv')
            scall = SkyCoord(csvdata['radeg'],csvdata['decdeg'],unit=u.deg)

        for name,ra,dec in zip(namelist,ralist,declist):
            # construct an ra/dec box search
            # PanSTARRS images are 3.1x3.1 deg, approximately
            if self.options.use_csv:
                csvdata = at.Table.read(self.options.csv_filename,format='ascii.csv')
                scall = SkyCoord(csvdata['radeg'],csvdata['decdeg'],unit=u.deg)
                sc = SkyCoord(ra,dec,unit=u.deg)
                iClose = np.where((sc.separation(scall).deg < 2.2) & (csvdata['diff_id'] != 'None'))[0]
                if iClose.size:
                    obs_data_dict[name] = {}
                    obs_data_dict[name]['ra'] = ra
                    obs_data_dict[name]['dec'] = dec
                    obs_data_dict[name]['results'] = [] #csvdata[iClose]
                    for c in csvdata[iClose]:
                        obs_data_dict[name]['results'] += [{
                            'survey_field':c['comment'],
                            'photometric_band':c['filter'][0],
                            'obs_mjd':date_to_mjd(c['dateobs']),
                            'image_id':c['exp_name'],
                            'diff_id':c['diff_id'],
                            'warp_id':c['warp_id'],
                            'stack_id':None}]
                        total_images += iClose.size
                else: print(f"Object {name} not found in YSE fields")

            else:
                ramin,ramax,decmin,decmax = getRADecBox(ra,dec,size=3.0)

                # check ra min/max are sensible, otherwise we have to be more clever
                if ramin > 0 and ramax < 360:
                    query_str = f"ra_gt={ramin}&ra_lt={ramax}&dec_gt={decmin}&dec_lt={decmax}&status_in=Successful&limit=100"
                    obs_data_response = requests.get(
                        f"{self.options.ysepzurl}/api/surveyobservations/?{query_str}",
                        auth=HTTPBasicAuth(self.options.ysepzuser,self.options.ysepzpass))

                    if obs_data_response.status_code != 200:
                        raise RuntimeError('issue communicating with the YSE-PZ server')
                    obs_data = obs_data_response.json()
                    obs_data_results = obs_data['results']

                elif ramin < 0:
                    ramin += 360
                    # might have to do two queries
                    query_str_1 = f"ra_gt={ramin}&dec_gt={decmin}&dec_lt={decmax}&status_in=Successful&limit=100"
                    query_str_2 = f"ra_lt={ramax}&dec_gt={decmin}&dec_lt={decmax}&status_in=Successful&limit=100"

                    obs_data_response_1 = requests.get(
                        f"{self.options.ysepzurl}/api/surveyobservations/?{query_str_1}",
                        auth=HTTPBasicAuth(self.options.ysepzuser,self.options.ysepzpass))
                    obs_data_response_2 = requests.get(
                        f"{self.options.ysepzurl}/api/surveyobservations/?{query_str_2}",
                        auth=HTTPBasicAuth(self.options.ysepzuser,self.options.ysepzpass))

                    if obs_data_response_1.status_code != 200 or obs_data_response_2.status_code != 200:
                        raise RuntimeError('issue communicating with the YSE-PZ server')
                    obs_data_1 = obs_data_response_1.json()
                    obs_data_2 = obs_data_response_2.json()
                    obs_data_results = np.append(obs_data_1['results'],obs_data_2['results'])

                obs_data_dict[name] = {}
                obs_data_dict[name]['ra'] = ra
                obs_data_dict[name]['dec'] = dec
                obs_data_dict[name]['results'] = obs_data_results
                total_images += len(obs_data_results)

                # we have to loop through because we have a 100-image limit here
                if len(obs_data_results) == 1000:
                    raise RuntimeWarning("""
There are more than 1000 images containing this coordinate!
Somebody needs to write a smarter code than this one to parse through all the data.
For now, doing first 1000 images only.
""")

        if total_images == 0:
            print('No images were found')
            return
            
        ### 2. stamp images & photometry ###
        phot_request_names,skycelldict,obs_table,status = self.request_forcedphot(obs_data_dict)
        if status: print('Images are all missing diff IDs'); return
        #for prn in phot_request_names: print(prn)
        
        ### 2c) check status of the stamp images/photometry jobs ###
        max_time = 30
        print(f'Jobs were submitted, waiting up to {max_time} minutes for them to finish...')
        # wait until the jobs are done
        jobs, jobs_done = len(phot_request_names), 0
        tstart = time.time()
        while jobs_done < jobs and time.time()-tstart < max_time*60:
            #print('waiting 60 seconds to check status...')
            time.sleep(60)
            jobs_done = 0
            for phot_request_name in phot_request_names:
                done_phot,success_phot = self.get_status(phot_request_name)
                jobs_done += done_phot
            print(f"Requesting images: {jobs_done} out of {jobs} done", end='\r')
        print(f"Requesting images: {jobs_done} out of {jobs} done")

        if jobs_done < jobs:
            raise RuntimeError('job timeout!')

        ### 2d) download the stamp images
        # save the data
        phot_dict = self.get_phot(phot_request_names,obs_table['transient_id','ra','dec'])

        ### 3. template images
        ### 3a) request the templates
        for i,k in enumerate(phot_dict.keys()):
            key_table = at.Table(phot_dict[k])['stack_id','ra','dec','camera']
            key_table['transient_id'] = [k] * len(key_table)
            phot_table = at.vstack([phot_table,key_table]) if i else key_table.copy()

        stack_table = at.Table(np.unique(phot_table))
        #stack_unq,idx = np.unique(phot_table['stack_id'],return_index=True)
        #stack_table = phot_table[idx]
        stack_request_name,skycelldict = self.request_templates(stack_table,skycelldict=skycelldict)
        print(f'Submitted stack request {stack_request_name}')

        ### 3b) check status of the templates
        job_done = False
        tstart = time.time()
        while not job_done and time.time()-tstart < max_time*60:
            #print('waiting 60 seconds to check status...')
            time.sleep(60)
            done_stamp,success_stamp = self.get_status(stack_request_name)
            job_done = done_stamp
        if not success_stamp: pass
        if not job_done: raise RuntimeError('job timeout!')

        ### 3c) download the stamp images
        print("Downloading stamp images...")
        stack_img_dict = self.get_stamps(stack_request_name, stack_table['stack_id','ra','dec'])
        gal_flux_dict = self.get_gal_flux(stack_img_dict)
        self.write_photometry(phot_dict, gal_flux_dict)        

        return
     
if __name__ == "__main__":
    usagestr = """
Tool for getting YSE (IPP) forced photometry for a given coordinate or set of coordinates.
Takes an individual name/RA/dec (decimal degrees) or a list of name/RA/dec (comma-delimited, ra/dec in decimal degrees).
    Having a unique name for each ra/dec coord is just for bookkeeping purposes.
    
Usage:
    python YSE_Forced_Position.py -n <name> -r <ra> -d <dec> -u <YSE-PZ username> -p <YSE-PZ password>
    python YSE_Forced_Position.py -c <coordinate list> -u <YSE-PZ username> -p <YSE-PZ password>
"""
    

    ys = YSE_Forced_Pos()

    parser = ys.add_args(usage=usagestr)
    args = parser.parse_args()
    ys.options = args

    if ys.options.coordlist is None and (ys.options.ra is None or ys.options.dec is None or ys.options.name is None):
        raise RuntimeError("""
must specify name/ra/dec or coordlist arguments, run:
        YSE_Forced_Position.py --help
for more info
""")
        
    if ys.options.coordlist is not None and not os.path.exists(ys.options.coordlist):
        raise RuntimeError(f'coordinate list file {ys.options.coordlist} does not exist')
    
    ys.main()
