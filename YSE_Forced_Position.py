#!/usr/bin/env python
# D. Jones - 6/21/23
### Query IPP forced photometry for any given ra,dec

import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import argparse
import os
from astropy.io import fits
from astropy.visualization import PercentileInterval, AsinhStretch
import astropy.table as at
import time
from io import BytesIO
import re
from lxml import html
from astropy.coordinates import SkyCoord
import astropy.units as u
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
def fits_to_png(ff,outfile,log=False):
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
    plt.savefig(outfile,bbox_inches = 'tight',pad_inches = 0)
    
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
 
        parser.add_argument(
            '--ysepzurl',type=str,default='https://ziggy.ucolick.org/yse',
            help='base URL for YSE-PZ, so you can reach the API and get field IDs')
        parser.add_argument(
            '-u','--ysepzuser',type=str,default=None,
            help='YSE-PZ username')
        parser.add_argument(
            '-p','--ysepzpass',type=str,default=None,
            help='YSE-PZ password')
        parser.add_argument(
            '--ifauser',type=str,default='ps1sc',
            help='IfA username')
        parser.add_argument(
            '--ifapass',type=str,default='skysurveys',
            help='IfA password')

        return parser

    def stamp_request(
            self,obs_data_dict,width=300,height=300,skycelldict=None,stack=False,submitrequest=True):

        transient_list,ra_list,dec_list,diff_id_list,warp_id_list,stack_id_list,camera_list = \
            np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
        for k in obs_data_dict.keys():
            for i,r in enumerate(obs_data_dict[k]['results']):
                if r['diff_id'] is None: continue
                transient_list = np.append(transient_list,k)
                ra_list = np.append(ra_list,obs_data_dict[k]['ra'])
                dec_list = np.append(dec_list,obs_data_dict[k]['dec'])    
                diff_id_list = np.append(diff_id_list,r['diff_id'])
                warp_id_list = np.append(warp_id_list,r['image_id'])
                stack_id_list = np.append(stack_id_list,r['stack_id'])
                camera_list = np.append(camera_list,get_camera(r['image_id']))
        if not len(transient_list):
            return None,{},transient_list,ra_list,dec_list,status
        
        data = at.Table(names=('ROWNUM', 'PROJECT', 'SURVEY_NAME', 'IPP_RELEASE', 'JOB_TYPE',
                               'OPTION_MASK', 'REQ_TYPE', 'IMG_TYPE', 'ID', 'TESS_ID',
                               'COMPONENT', 'COORD_MASK', 'CENTER_X', 'CENTER_Y', 'WIDTH',
                               'HEIGHT', 'DATA_GROUP', 'REQFILT', 'MJD_MIN', 'MJD_MAX',
                               'RUN_TYPE', 'FWHM_MIN', 'FWHM_MAX', 'COMMENT'),
                        dtype=('>i4','S16','S16','S16','S16','>i4','S16','S16','S16','S64',
                               'S64','>i4','>f8','>f8','>f8','>f8','S64','S16','>f8','>f8',
                               'S16','>f8','>f8','S64'))

        if skycelldict is None:
            transient_unq,idx = np.unique(transient_list,return_index=True)
            skycelldict = {}
            skycells = np.array([])
            for snid,ra,dec in zip(transient_list[idx],ra_list[idx],dec_list[idx]):
                skycells = np.append(skycells,'skycell.'+getskycell(ra,dec)[0])
                skycelldict[snid] = skycells[-1]
        
        count = 1
        for snid,ra,dec,camera,diff_id in \
            zip(transient_list,ra_list,dec_list,camera_list,diff_id_list):
            if diff_id is None: continue
            skycell_str = skycelldict[snid]
            data.add_row((count,camera,'null','null','stamp',2049,'byid','diff',diff_id,'RINGS.V3',
                          skycell_str,2,ra,dec,width,height,'null','null',0,0,'null',0,0,'diff.for.%s'%snid) )
            count += 1
        
        for snid,ra,dec,camera,warp_id in \
            zip(transient_list,ra_list,dec_list,camera_list,warp_id_list):
            if warp_id is None: continue
            skycell_str = skycelldict[snid]
            data.add_row((count,camera,'null','null','stamp',2049,'byid','warp',warp_id,'RINGS.V3',
                          skycell_str,2,ra,dec,width,height,'null','null',0,0,'null',0,0,'warp.for.%s'%snid) )
            count += 1
            
        for snid,ra,dec,camera,stack_id in \
            zip(transient_list,ra_list,dec_list,camera_list,stack_id_list):
            if stack_id is None: continue
            skycell_str = skycelldict[snid]
            data.add_row((count,camera,'null','null','stamp',2049,'byid','stack',stack_id,'RINGS.V3',
                          skycell_str,2,ra,dec,width,height,'null','null',0,0,'null',0,0,'stack.for.%s'%snid) )
            count += 1
        if submitrequest:
            hdr = default_stamp_header.copy()
            request_name = 'YSE-stamp.%i'%(time.time())
            hdr['REQ_NAME'] = request_name
            ff = fits.BinTableHDU(data, header=hdr)

            s = BytesIO()
            ff.writeto(s, overwrite=True)

            self.submit_to_ipp(s)
            return request_name,skycelldict,transient_list,ra_list,dec_list,0
        else:
            return None,skycelldict,transient_list,ra_list,dec_list,0
        
    def stamp_request_stack(
            self,transient_list,ra_list,dec_list,camera_list,diff_id_list,
            warp_id_list,stack_id_list,width=300,height=300,skycelldict=None):

        assert len(transient_list) == len(warp_id_list) or \
            len(transient_list) == len(diff_id_list) or \
            len(transient_list) == len(stack_id_list)
        
        transient_list,ra_list,dec_list,diff_id_list,warp_id_list,stack_id_list = \
            np.atleast_1d(np.asarray(transient_list)),np.atleast_1d(np.asarray(ra_list)),np.atleast_1d(np.asarray(dec_list)),\
            np.atleast_1d(np.asarray(diff_id_list)),np.atleast_1d(np.asarray(warp_id_list)),np.atleast_1d(np.asarray(stack_id_list))
        
        data = at.Table(names=('ROWNUM', 'PROJECT', 'SURVEY_NAME', 'IPP_RELEASE', 'JOB_TYPE',
                               'OPTION_MASK', 'REQ_TYPE', 'IMG_TYPE', 'ID', 'TESS_ID',
                               'COMPONENT', 'COORD_MASK', 'CENTER_X', 'CENTER_Y', 'WIDTH',
                               'HEIGHT', 'DATA_GROUP', 'REQFILT', 'MJD_MIN', 'MJD_MAX',
                               'RUN_TYPE', 'FWHM_MIN', 'FWHM_MAX', 'COMMENT'),
                        dtype=('>i4','S16','S16','S16','S16','>i4','S16','S16','S16','S64',
                               'S64','>i4','>f8','>f8','>f8','>f8','S64','S16','>f8','>f8',
                               'S16','>f8','>f8','S64'))

        transients_unq,idx = np.unique(transient_list,return_index=True)
        if skycelldict is None:
            skycelldict = {}
            skycells = np.array([])
            for snid,ra,dec in zip(transient_list[idx],ra_list[idx],dec_list[idx]):
                skycells = np.append(skycells,'skycell.'+getskycell(ra,dec)[0])
                skycelldict[snid] = skycells[-1]
        
        count = 1
        for snid,ra,dec,camera,diff_id in \
            zip(transient_list,ra_list,dec_list,camera_list,diff_id_list):
            if diff_id is None: continue
            skycell_str = skycelldict[snid]
            data.add_row((count,camera,'null','null','stamp',2049,'byid','diff',diff_id,'RINGS.V3',
                          skycell_str,2,ra,dec,width,height,'null','null',0,0,'null',0,0,'diff.for.%s'%snid) )
            count += 1
        
        for snid,ra,dec,camera,warp_id in \
            zip(transient_list,ra_list,dec_list,camera_list,warp_id_list):
            if warp_id is None: continue
            skycell_str = skycelldict[snid]
            data.add_row((count,camera,'null','null','stamp',2049,'byid','warp',warp_id,'RINGS.V3',
                          skycell_str,2,ra,dec,width,height,'null','null',0,0,'null',0,0,'warp.for.%s'%snid) )
            count += 1
            
        for snid,ra,dec,camera,stack_id in \
            zip(transient_list,ra_list,dec_list,camera_list,stack_id_list):
            if stack_id is None: continue
            skycell_str = skycelldict[snid]
            data.add_row((count,camera,'null','null','stamp',2049,'byid','stack',stack_id,'RINGS.V3',
                          skycell_str,2,ra,dec,width,height,'null','null',0,0,'null',0,0,'stack.for.%s'%snid) )
            count += 1

        hdr = default_stamp_header.copy()
        request_name = 'YSE-stamp.%i'%(time.time())
        hdr['REQ_NAME'] = request_name
        ff = fits.BinTableHDU(data, header=hdr)

        s = BytesIO()
        ff.writeto(s, overwrite=True)
        #if self.debug:
        #    ff.writeto('stampimg.fits',overwrite=True)

        self.submit_to_ipp(s)
        return request_name,skycelldict

    
    def forcedphot_request(self,obs_data_dict,skycelldict):

        transient_list,ra_list,dec_list,mjd_list,filt_list,diff_id_list,warp_id_list,stack_id_list,camera_list = \
            np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
        for k in obs_data_dict.keys():
            for i,r in enumerate(obs_data_dict[k]['results']):
                if r['diff_id'] is None: continue
                transient_list = np.append(transient_list,k)
                ra_list = np.append(ra_list,obs_data_dict[k]['ra'])
                dec_list = np.append(dec_list,obs_data_dict[k]['dec'])
                mjd_list = np.append(mjd_list,r['obs_mjd'])
                # note this line is hacky
                if not self.options.use_csv:
                    filt_list = np.append(filt_list,phot_band_dict[r['photometric_band']])
                else:
                    filt_list = np.append(filt_list,r['photometric_band'])
                diff_id_list = np.append(diff_id_list,r['diff_id'])
                warp_id_list = np.append(warp_id_list,r['warp_id'])
                stack_id_list = np.append(stack_id_list,r['stack_id'])
                camera_list = np.append(camera_list,get_camera(r['image_id']))
        if not len(transient_list):
            return [],1

        
        request_names = []
        count = 0
        for snid_unq in np.unique(transient_list):
            data = at.Table(names=('ROWNUM','PROJECT','RA1_DEG','DEC1_DEG','RA2_DEG','DEC2_DEG','FILTER','MJD-OBS','FPA_ID','COMPONENT_ID'),
                            dtype=('S20','S16','>f8','>f8','>f8','>f8','S20','>f8','>i4','S64'))
            for snid,ra,dec,mjd,filt,camera,diff_id in \
                zip(transient_list[transient_list == snid_unq],ra_list[transient_list == snid_unq],dec_list[transient_list == snid_unq],
                    mjd_list[transient_list == snid_unq],filt_list[transient_list == snid_unq],camera_list[transient_list == snid_unq],diff_id_list[transient_list == snid_unq]):
                if diff_id is None or diff_id == 'NULL': continue
                data.add_row(('forcedphot_ysebot_{}'.format(count),camera,ra,dec,ra,dec,filt,mjd,diff_id,skycelldict[snid_unq]) )
                count += 1

            if len(data) > 0:
                hdr = default_forcedphot_header.copy()
                request_name = 'YSE-phot.%s.%s.%i'%(snid,diff_id,time.time())
                hdr['QUERY_ID'] = request_name
                hdr['EXTNAME'] = 'MOPS_DETECTABILITY_QUERY'
                hdr['EXTVER'] = '2'
                hdr['OBSCODE'] = '566'
                hdr['STAGE'] = 'WSdiff'
                ff = fits.BinTableHDU(data, header=hdr)
                #if self.debug:
                #    ff.writeto('%s.fits'%request_name, overwrite=True)

                s = BytesIO()
                ff.writeto(s, overwrite=True)

                self.submit_to_ipp(s)
                request_names += [request_name]
            else:
                print('warning : no diff IDs for transient {}'.format(snid_unq))
                
        return request_names,0

    def get_phot(self,request_names,transient_list,transient_ra_list,transient_dec_list,img_dict):
        sct = SkyCoord(transient_ra_list,transient_dec_list,unit=u.deg)
        transient_list = np.array(transient_list)

        phot_dict = {}
        for request_name in request_names:
            phot_link = 'http://datastore.ipp.ifa.hawaii.edu/pstampresults/'
            phot_results_link = '{}/{}/'.format(phot_link,request_name)

            phot_page = requests.get(url=phot_results_link)
            if phot_page.status_code != 200:
                raise RuntimeError('results page {} does not exist'.format(phot_results_link))


            tree = html.fromstring(phot_page.content)
            fitsfiles = tree.xpath('//a/text()')
            for f in fitsfiles:
                if 'detectability' in f:
                    phot_fits_link = '{}/{}/{}'.format(phot_link,request_name,f)
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
                        stack_id = ff[0].header['PPSUB.REFERENCE'].split('.')[-3]
                        warp_id = ff[0].header['PPSUB.INPUT'].split('.')[3]
                        ra = ff[1].data['RA_PSF'][i]
                        dec = ff[1].data['DEC_PSF'][i]
                        sc = SkyCoord(ff[1].data['RA_PSF'][i],ff[1].data['DEC_PSF'][i],unit=u.deg)
                        sep = sc.separation(sct).arcsec
                        if np.min(sep) > 2:
                            raise RuntimeError(
                                'couldn\'t find transient match for RA,Dec=%.7f,%.7f'%(
                                    ff[1].data['RA_PSF'][i],ff[1].data['DEC_PSF'][i]))
                        tn = transient_list[sep == np.min(sep)][0]
                        if tn not in phot_dict.keys():
                            phot_dict[tn] = {'mjd':[],
                                             'filt':[],
                                             'flux':[],
                                             'flux_err':[],
                                             'dq':[],
                                             'stack_id':[],
                                             'warp_id':[],
                                             'diff_id':[],
                                             'ra':[],
                                             'dec':[],
                                             'exptime':[],
                                             'zpt':[],
                                             'camera':[]}

                        phot_dict[tn]['mjd'] += [mjd]
                        phot_dict[tn]['filt'] += [filt]
                        phot_dict[tn]['flux'] += [flux]
                        phot_dict[tn]['flux_err'] += [flux_err]
                        phot_dict[tn]['dq'] += [dq]
                        phot_dict[tn]['stack_id'] += [stack_id]
                        phot_dict[tn]['warp_id'] += [warp_id]
                        phot_dict[tn]['diff_id'] += [f.split('.')[2]]
                        phot_dict[tn]['ra'] += [ra]
                        phot_dict[tn]['dec'] += [dec]
                        phot_dict[tn]['exptime'] += [exptime]
                        phot_dict[tn]['zpt'] += [ff[0].header['FPA.ZP']]
                        phot_dict[tn]['camera'] += [ff[0].header['FPA.INSTRUMENT']]

        return phot_dict

    def write_photometry(self,phot_dict):

        for t in phot_dict.keys():
            with open(f"{self.options.outdir}/{t}_phot.dat",'w') as fout:
                print('# mjd filt flux flux_err dq',file=fout)
                for m,f,fx,fxe,dq in zip(phot_dict[t]['mjd'],phot_dict[t]['filt'],phot_dict[t]['flux'],
                                         phot_dict[t]['flux_err'],phot_dict[t]['dq']):
                    print(f"{m:.3f} {f} {fx:.4f} {fxe:.4f} {dq}",file=fout)
        
        return
    
    def get_stamps(self,request_name,transient_list):

        stamp_link = 'http://datastore.ipp.ifa.hawaii.edu/yse-pstamp-results/'
        stamp_results_link = '{}/{}/results.mdc'.format(stamp_link,request_name)
        stamp_fitsfile_link = '{}/{}/'.format(stamp_link,request_name)
        
        stamps_page = requests.get(url=stamp_results_link)
        if stamps_page.status_code != 200:
            raise RuntimeError('results page {} does not exist'.format(stamp_results_link))
        
        mdc_stamps = parse_mdc(stamps_page.text)

        tree = html.fromstring(stamps_page.content)
        fitsfiles = tree.xpath('//a/text()')
        
        image_dict = {}

        for k in mdc_stamps.keys():
            if 'SUCCESS' not in mdc_stamps[k]['ERROR_STR'] and 'NO_VALID_PIXELS' not in mdc_stamps[k]['ERROR_STR'] and\
               'PSTAMP_NO_IMAGE_MATCH' not in mdc_stamps[k]['ERROR_STR']:
                print('warning: part of job {} failed!'.format(request_name))
            else:
                img_name,img_type,transient,mjd,img_id,img_filter,img_camera = \
                    mdc_stamps[k]['IMG_NAME'],mdc_stamps[k]['IMG_TYPE'],mdc_stamps[k]['COMMENT'].split('.')[-1],\
                    float(mdc_stamps[k]['MJD_OBS']),mdc_stamps[k]['ID'],mdc_stamps[k]['FILTER'].split('.')[0],\
                    mdc_stamps[k]['PROJECT']

                if transient not in image_dict.keys():
                    image_dict[transient] = {'warp_image_link':[],'diff_image_link':[],'stack_image_link':[],
                                             'warp_image_id':[],'diff_image_id':[],'stack_image_id':[],
                                             'warp_image_mjd':[],'diff_image_mjd':[],'stack_image_mjd':[],
                                             'warp_image_filter':[],'diff_image_filter':[],'stack_image_filter':[],
                                             'warp_image_camera':[],'diff_image_camera':[],'stack_image_camera':[]}
                if 'NO_VALID_PIXELS' not in mdc_stamps[k]['ERROR_STR'] and 'PSTAMP_NO_IMAGE_MATCH' not in mdc_stamps[k]['ERROR_STR']:
                    if img_type == 'warp':
                        image_dict[transient]['warp_image_link'] += ['{}/{}'.format(stamp_fitsfile_link,img_name)]
                        image_dict[transient]['warp_image_id'] += [img_id]
                        image_dict[transient]['warp_image_mjd'] += [mjd]
                        image_dict[transient]['warp_image_filter'] += [img_filter]
                        image_dict[transient]['warp_image_camera'] += [img_camera]
                    elif img_type == 'diff':
                        image_dict[transient]['diff_image_link'] += ['{}/{}'.format(stamp_fitsfile_link,img_name)]
                        image_dict[transient]['diff_image_id'] += [img_id]
                        image_dict[transient]['diff_image_mjd'] += [mjd]
                        image_dict[transient]['diff_image_filter'] += [img_filter]
                        image_dict[transient]['diff_image_camera'] += [img_camera]
                    elif img_type == 'stack':
                        image_dict[transient]['stack_image_link'] += ['{}/{}'.format(stamp_fitsfile_link,img_name)]
                        image_dict[transient]['stack_image_id'] += [img_id]
                        image_dict[transient]['stack_image_mjd'] += [mjd]
                        image_dict[transient]['stack_image_filter'] += [img_filter]
                        image_dict[transient]['diff_image_camera'] += [img_camera]
                    else: raise RuntimeError('image type {} not found'.format(img_type))
                else:
                    if img_type == 'warp':
                        image_dict[transient]['warp_image_link'] += [None]
                        image_dict[transient]['warp_image_id'] += [img_id]
                        image_dict[transient]['warp_image_mjd'] += [mjd]
                        image_dict[transient]['warp_image_filter'] += [img_filter]
                        image_dict[transient]['warp_image_camera'] += [img_camera]
                    elif img_type == 'diff':
                        image_dict[transient]['diff_image_link'] += [None]
                        image_dict[transient]['diff_image_id'] += [img_id]
                        image_dict[transient]['diff_image_mjd'] += [mjd]
                        image_dict[transient]['diff_image_filter'] += [img_filter]
                        image_dict[transient]['diff_image_camera'] += [img_camera]
                    elif img_type == 'stack':
                        image_dict[transient]['stack_image_link'] += [None]
                        image_dict[transient]['stack_image_id'] += [img_id]
                        image_dict[transient]['stack_image_mjd'] += [mjd]
                        image_dict[transient]['stack_image_filter'] += [img_filter]
                        image_dict[transient]['stack_image_camera'] += [img_camera]
                    else: raise RuntimeError('image type {} not found'.format(img_type))

        return image_dict

    def get_status(self,request_name):
        
        status_link = 'http://pstamp.ipp.ifa.hawaii.edu/status.php'
        session = requests.Session()
        session.auth = (self.options.ifauser,self.options.ifapass)

        page = session.post(status_link)
        page = session.post(status_link)
        
        if page.status_code == 200:
            lines_out = []
            for line in page.text.split('<pre>')[-1].split('\n'):
                if line and '------------------' not in line and '/pre' not in line:
                    lines_out += [line[1:]]
            text = '\n'.join(lines_out)
            tbl = at.Table.read(text,format='ascii',delimiter='|',data_start=1,header_start=0)

            idx = tbl['name'] == request_name
            if not len(tbl[idx]):
                print('warning: could not find request named %s'%request_name)
                return False, False
            if tbl['Completion Time (UTC)'][idx]: done = True
            else: done = False

            if float(tbl['Total Jobs'][idx]) == float(tbl['Successful Jobs'][idx]): success = True
            else:
                success = False
                print('warning: %i of %i jobs failed'%(float(tbl['Total Jobs'][idx])-float(tbl['Successful Jobs'][idx]),float(tbl['Total Jobs'][idx])))
        
        return done,success

    def get_images(self,img_dict):

        basedir = self.options.outdir
        for k in img_dict.keys():
            for img_key_in,img_key_out in zip(['warp_image_link','diff_image_link','stack_image_link'],
                                              ['warp_file','diff_file','stack_file']):

                img_dict[k][img_key_out] = []
                img_dict[k][img_key_out+'_png'] = []

                session = requests.Session()
                session.auth = (self.options.ifauser,self.options.ifapass)
                for i in range(len(img_dict[k][img_key_in])):
                    if img_dict[k][img_key_in][i] is None:
                        img_dict[k][img_key_out] += [None]
                        img_dict[k][img_key_out+'_png'] += [None]
                        continue
                    
                    outdir = "%s/%s/%i"%(basedir,k,int(float(img_dict[k]['%s_mjd'%img_key_in.replace('_link','')][i])))
                    
                    if not os.path.exists(outdir):
                        os.makedirs(outdir)

                    filename = img_dict[k][img_key_in][i].split('/')[-1]
                    outfile = "{}/{}".format(outdir,filename)
                    
                    fits_response = session.get(img_dict[k][img_key_in][i],stream=True)
                    with open(outfile,'wb') as fout:
                        shutil.copyfileobj(fits_response.raw, fout)

                    ff = fits.open(outfile)
                    if 'diff' in img_key_in: fits_to_png(ff,outfile.replace('fits','png'),log=False)
                    else: fits_to_png(ff,outfile.replace('fits','png'),log=True)
                    img_dict[k][img_key_out] += [outfile.replace('{}/'.format(basedir),'')]
                    img_dict[k][img_key_out+'_png'] += [outfile.replace('{}/'.format(basedir),'').replace('.fits','.png')]

        return img_dict

    def get_all_images(self,img_dict,stack_img_dict):

        img_dict = self.get_images(img_dict)
        stack_img_dict = self.get_images(stack_img_dict)

        return
    
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
            self.all_stages([self.options.name],[self.options.ra],[self.options.dec])
            if self.options.checkfiles and os.path.exists(f'{self.options.outdir}/{self.options.name}_phot.dat'): pass
            else: self.all_stages([self.options.name],[self.options.ra],[self.options.dec])
        elif self.options.coordlist:
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
        for name,ra,dec in zip(namelist,ralist,declist):
            # construct an ra/dec box search
            # PanSTARRS images are 3.1x3.1 deg, approximately
            if not self.options.use_csv:
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
            else:
                csvdata = at.Table.read(self.options.csv_filename,format='ascii.csv')
                scall = SkyCoord(csvdata['radeg'],csvdata['decdeg'],unit=u.deg)
                sc = SkyCoord(ra,dec,unit=u.deg)
                iClose = np.where((sc.separation(scall).deg < 2.2) & (csvdata['diff_id'] != 'None'))[0]
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
                    total_images = len(iClose)
                
            # we have to loop through because we have a 100-image limit here
            if not self.options.use_csv and len(obs_data_results) == 1000:
                   raise RuntimeWarning("""
There are more than 1000 images containing this coordinate!
Somebody needs to write a smarter code than this one to parse through all the data.
For now, doing first 1000 images only.
""")

        if total_images == 0:
            print('No images were found')
            return
        
        ### 2. stamp images & photometry ###
        ### 2a) request the stamp images
        if not self.options.nofitsdownload:
            stamp_request_name,skycelldict,transient_list,ra_list,dec_list,status = self.stamp_request(
                obs_data_dict)
            if status:
                print('Images are all missing diff/warp IDs')
                return
            print('submitted stamp request {}'.format(stamp_request_name))
        else:
            _,skycelldict,transient_list,ra_list,dec_list,status = self.stamp_request(
                obs_data_dict,submitrequest=False)
        ### 2b) request the photometry
        phot_request_names,status = self.forcedphot_request(
            obs_data_dict,skycelldict)
        if self.options.nofitsdownload and status:
            print('No images were found')
            return
            
        print('submitted phot requests:')
        for prn in phot_request_names: print(prn)

        ### 2c) check status of the stamp images/photometry jobs ###
        print('jobs were submitted, waiting up to 25 minutes for them to finish')
        # wait until the jobs are done
        jobs_done = False
        tstart = time.time()
        while not jobs_done and time.time()-tstart < 1500:
            print('waiting 60 seconds to check status...')
            time.sleep(60)
            if not self.options.nofitsdownload:
                done_stamp,success_stamp = self.get_status(stamp_request_name)
            doneall_phot = True
            for phot_request_name in phot_request_names:
                done_phot,success_phot = self.get_status(phot_request_name)
                if not done_phot: doneall_phot = False
            if not self.options.nofitsdownload and done_stamp and doneall_phot: jobs_done = True
            elif self.options.nofitsdownload and doneall_phot: jobs_done = True

            
        if not jobs_done:
            raise RuntimeError('job timeout!')

        ### 2d) download the stamp images
        if not self.options.nofitsdownload:
            img_dict = self.get_stamps(stamp_request_name,transient_list)

        # save the data
        phot_dict = \
            self.get_phot(phot_request_names,transient_list,ra_list,dec_list,{})
        self.write_photometry(phot_dict)

        ### 3. template images
        ### 3a) request the templates
        if not self.options.nofitsdownload:
            transient_list,ra_list,dec_list,stack_id_list,camera_list = [],[],[],[],[]
            for t in phot_dict.keys():
                for s,r,d,c in zip(phot_dict[t]['stack_id'],phot_dict[t]['ra'],phot_dict[t]['dec'],phot_dict[t]['camera']):
                    stack_id_list += [s]
                    ra_list += [r]
                    dec_list += [d]
                    transient_list += [t]
                    camera_list += [c]

            stack_request_name,skycelldict = self.stamp_request_stack(
                transient_list,ra_list,dec_list,camera_list,[],[],stack_id_list,skycelldict=skycelldict)
            print('submitted stack request {}'.format(stack_request_name))

            ### 3b) check status of the templates
            tstart = time.time()
            jobs_done = False
            while not jobs_done and time.time()-tstart < 900:
                print('waiting 60 seconds to check status...')
                time.sleep(60)
                done_stamp,success_stamp = self.get_status(stack_request_name)
                if done_stamp: jobs_done = True
            if not success_stamp:
                pass

            if not jobs_done:
                raise RuntimeError('job timeout!')

            ### 3c) download the stamp images
            stack_img_dict = self.get_stamps(stack_request_name,transient_list)
            self.get_all_images(img_dict,stack_img_dict)
        
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

