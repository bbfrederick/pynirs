#!/usr/bin/env python
#
#       $Author: frederic $
#       $Date: 2016/09/20 20:47:15 $
#       $Id: spo2funcs.py,v 1.9 2016/09/20 20:47:15 frederic Exp $
#
import bisect
import os
import sys

import numpy as np
import pandas as pd
import pylab as plt
import pywt
import rapidtide.filter as tide_filt
from scipy import integrate, interpolate
from scipy.optimize import leastsq
from scipy.signal import cspline1d, cspline1d_eval, firwin
from scipy.special import erfinv
from statsmodels.robust import mad


def dependencies_for_hemofuncs():
    from scipy.sparse.csgraph import _validation


# define some important globals here
coeff_scale = 16
MAXLINES = 1000000000
stage2gain = 15.0
searchrange = 10.0
hbthresh = 1.0
numhbstoavg = 5
scopetype = 0
adjustleds = False
waitforrq = False
aafilt = False
pwrlnfilt = False
lowpower = False
trapezoidalfftfilter = True
physioFIRtime = 3.0  # now specified in seconds
acdcFIRtime = 2.0  # now specified in seconds
QRSFIRtime = 1.0  # now specified in seconds
vlf_upperpass = 0.009
vlf_upperstop = 0.010

lf_lowerstop = vlf_upperpass
lf_lowerpass = vlf_upperstop
lf_upperpass = 0.15
lf_upperstop = 0.20

resp_lowerstop = lf_upperpass
resp_lowerpass = lf_upperstop
resp_upperpass = 0.4
resp_upperstop = 0.5

card_lowerstop = resp_upperpass
card_lowerpass = resp_upperstop
card_upperpass = 4.0
card_upperstop = 5.0

cardsmooth_upperpass = 0.4
cardsmooth_upperstop = 0.5

global physioFIRinited
global acdcFIRinited
global QRSFIRinited
physioFIRinited = False
acedFIRinited = False
QRSFIRinited = False
global smoothFIRinited
smoothFIRinited = False
defaultbutterorder = 3

# initialize dictionaries
eHbO = {}
eHb = {}
dpf = {}

eHbO["460"] = 0.3346  # HbO @ 660
eHb["460"] = 3.4408  # Hb @ 660
dpf["460"] = 6.0  # 660 nm (assumed, based on conversation with Angelo)

eHbO["523"] = 0.3346  # HbO @ 660
eHb["523"] = 3.4408  # Hb @ 660
dpf["523"] = 6.0  # 660 nm (assumed, based on conversation with Angelo)

# extinction coefficients from Mark Cope's 1991 PhD Thesis
eHbO["660"] = 0.3346  # HbO @ 660
eHb["660"] = 3.4408  # Hb @ 660
dpf["660"] = 6.0  # 660 nm (assumed, based on conversation with Angelo)

eHbO["780"] = 0.7360
eHb["780"] = 1.1050

eHbO["830"] = 1.0507
eHb["830"] = 0.7804

eHbO["850"] = 1.1596  # HbO @ 850 - use for PluX headband
eHb["850"] = 0.7861  # HbO @ 850 - use for PluX headband
# sdspace = 2.0  		# source detector spacing - PluX headband
dpf["850"] = (
    5.5  # 850 nm (estimated from "General equation for the differential pathlength factor of the frontal human head depending on wavelength and age" by Scholkmann)
)

eHbO["880"] = 1.14  # HbO @ 880 - use for Medwatch
eHb["880"] = 0.83  # HbO @ 880 - use for Medwatch
dpf["880"] = 5.5  # DPF @ 880 - Medwatch


eHbO["920"] = 1.3590  # HbO @ 920 - use for ADAPTER
eHb["920"] = 0.8844  # HbO @ 920 - use for ADAPTER
dpf["920"] = (
    5.5  # 850 nm (estimated from "General equation for the differential pathlength factor of the frontal human head depending on wavelength and age")
)

eHbO["950"] = 1.3374  # HbO @ 950 - use for PluX armband
eHb["950"] = 0.7068  # HbO @ 950 - use for PluX armband
# sdspace = 2.83 		# source detector spacing - PluX armband
dpf["950"] = (
    5.5  # 950 nm (estimated from the 850 number, and the fact that normalized OD is about the same at 850 and 950
)
# according to "Determination of the wavelength dependence of the differential pathlength factor from near-infrared pulse signals." by Kohl")

# simulation parameters
sim_oxygenation = 0.95
sim_lfosize = 5.0
sim_cardsize = 5.0
sim_irvisfac = 2.0
sim_lfofreq = 0.1
sim_cardfreq = 1.0
sim_noiselevel = 1.0

cardminwindowlen = 100

alpha1 = 0.1
alpha2 = 0.5

simpacketnum = 0


# Design a cardiac rate smoothing lowpass FIR filter
def design_cardsmooth_filter(Fs, n):
    Fc = 0.4
    a = firwin(n, cutoff=Fc / Fs, window="hamming")
    return a


# Design a GSR lowpass FIR filter
def design_GSR_filter(Fs, n):
    Fc = 1.0
    a = firwin(n, cutoff=Fc / Fs, window="hamming")
    return a


# Design a LFO lowpass FIR filter
def design_LFO_filter(Fs, n):
    Fc = 0.2
    a = firwin(n, cutoff=Fc / Fs, window="hamming")
    return a


# Design a raw DC lowpass FIR filter
def design_DC_filter(Fs, n):
    Fc = 0.45
    a = firwin(n, cutoff=Fc / Fs, window="hamming")
    return a


# Design a cardiac bandpass FIR filter
def design_card_filter(Fs, n, cardratelow=40.0, cardratehigh=120.0, numharmonics=3):
    cardtimelow = 60.0 / cardratelow
    cardtimehigh = 60.0 / cardratehigh
    cardfreqlow = 1.0 / cardtimelow
    cardfreqhigh = (
        1.0 * numharmonics
    ) / cardtimehigh  # allow for first and second harmonics
    print("cardiac filter range:", cardfreqlow, " to ", cardfreqhigh, " Hz")
    Fcl = cardfreqlow
    Fch = cardfreqhigh
    a = firwin(n, cutoff=Fcl / Fs, window="blackmanharris")
    b = -firwin(n, cutoff=Fch / Fs, window="blackmanharris")
    b[int(n / 2)] += 1
    d = -(a + b)
    d[int(n / 2)] += 1
    return d


# Design a respiratory bandpass FIR filter
def design_resp_filter(Fs, n):
    respratelow = 12.0
    respratehigh = 20.0
    resptimelow = 60.0 / respratelow
    resptimehigh = 60.0 / respratehigh
    respfreqlow = 1.0 / resptimelow
    respfreqhigh = 1.0 / resptimehigh
    print("Respiratory filter range:", respfreqlow, " to ", respfreqhigh, " Hz")
    Fcl = respfreqlow
    Fch = respfreqhigh
    a = firwin(n, cutoff=Fcl / Fs, window="blackmanharris")
    b = -firwin(n, cutoff=Fch / Fs, window="blackmanharris")
    b[int(n / 2)] += 1
    d = -(a + b)
    d[int(n / 2)] += 1
    return d


# Design a QRS discriminator highpass FIR filter
def design_QRS_filter(Fs, n):
    Fcl = 2.0
    Fch = 150.0
    a = firwin(n, cutoff=Fcl / Fs, window="blackmanharris")
    b = -firwin(n, cutoff=Fch / Fs, window="blackmanharris")
    b[n / 2] += 1
    d = -(a + b)
    d[n / 2] += 1
    return d


# determine the maximum delay of all the filters
def maxdelay():
    global physioFIRtime
    # return np.max([cardsmoothfiltlen, lfofiltlen, cardfiltlen, cardminwindowlen])
    return physioFIRtime


def get_numhbstoavg():
    global numhbstoavg
    return numhbstoavg


# send a command to the shell
def doashellcmd(cmd):
    a = os.popen(cmd)
    while 1:
        line = a.readline()
        if not line:
            break
        retval = line[:-1]
        return retval


def sendcmdbyte(theport, thecmd):
    theport.write(thecmd)


peakwinlen = 6


def checkforpeak(thecarddata, cardmax, cardmin, thetimepoint, lastpeak):
    diffs = (
        thecarddata[thetimepoint - peakwinlen : thetimepoint]
        - thecarddata[thetimepoint - peakwinlen - 1 : thetimepoint - 1]
    )
    if (
        diffs[-1] < 0.0 < np.mean(diffs[-peakwinlen:-2])
        and thetimepoint - lastpeak > peakwinlen
        and thecarddata[thetimepoint] > cardmin + 3.0 * (cardmax - cardmin) / 4.0
    ):
        return True
    else:
        return False


def nirstohb(
    val1, val2, lambda1=780, lambda2=830, sdspace=1.0, dontcombine=False, age=20
):
    delta_OD_1 = -np.log(val1)
    delta_OD_2 = -np.log(val2)
    key_1 = str(lambda1)
    key_2 = str(lambda2)
    # vis_dp = vis_dpf[viskey] * sdspace  # calculate the differential path
    # ir_dp = ir_dpf[irkey] * sdspace  # calculate the differential path
    dp_1 = calculateDPF(lambda1, age=age)
    dp_2 = calculateDPF(lambda2, age=age)
    den = eHbO[key_1] * eHb[key_2] - eHbO[key_2] * eHb[key_1]

    if dontcombine:
        HbO_raw_data = 1000.0 * delta_OD_2 / dp_2
        HbR_raw_data = 1000.0 * delta_OD_1 / dp_1
        tHb_raw_data = HbO_raw_data + HbR_raw_data
    else:
        HbO_raw_data = (
            1000.0
            * (delta_OD_1 / dp_1 * eHb[key_2] - delta_OD_2 / dp_2 * eHb[key_1])
            / den
        )
        HbR_raw_data = (
            1000.0
            * (delta_OD_2 / dp_2 * eHbO[key_1] - delta_OD_1 / dp_1 * eHbO[key_2])
            / den
        )
        tHb_raw_data = HbO_raw_data + HbR_raw_data
    return HbO_raw_data, HbR_raw_data, tHb_raw_data


# @functools.lru_cache() need poython >3.2
def calculateDPF(wavelength, age=20):
    # https://www.ncbi.nlm.nih.gov/pubmed/24121731
    return (
        223.3
        + 0.05624 * (age) ** 0.8493
        + -5.723e-7 * (wavelength) ** 3
        + 0.001245 * (wavelength) ** 2
        + -0.9025 * (wavelength)
    )


""" Converts internsity (raw data) to optical density
INPUT
- irval : normalized ir values
- visval: normalized vis values
OUTPUT
- (delta_OD_ir, delta_OD_vis) : change in optical density tuple for IR and VIS values, respectively
"""


def intensity_to_OD(irval, visval):
    delta_OD_ir = np.where(irval > 0.0, -np.log(irval), 0.0)
    delta_OD_vis = np.where(visval > 0.0, -np.log(visval), 0.0)
    return (delta_OD_ir, delta_OD_vis)


def od_to_conc(delta_OD_ir, delta_OD_vis, vislambda=660, irlambda=920, sdspace=1.0):
    viskey = str(vislambda)
    irkey = str(irlambda)
    vis_dp = dpf[viskey] * sdspace  # calculate the differential path
    ir_dp = dpf[irkey] * sdspace  # calculate the differential path
    den = eHbO[viskey] * eHb[irkey] - eHbO[irkey] * eHb[viskey]

    HbO_raw_data = (
        1000.0
        * (delta_OD_vis / vis_dp * eHb[irkey] - delta_OD_ir / ir_dp * eHb[viskey])
        / den
    )
    HbR_raw_data = (
        1000.0
        * (delta_OD_ir / ir_dp * eHbO[viskey] - delta_OD_vis / vis_dp * eHbO[irkey])
        / den
    )
    tHb_raw_data = HbO_raw_data + HbR_raw_data
    return HbO_raw_data, HbR_raw_data, tHb_raw_data


def checkmaxmin(thenewpoint, thecurrentmax, thecurrentmin):
    if thenewpoint > thecurrentmax:
        thecurrentmax = thenewpoint
    if thenewpoint < thecurrentmin:
        thecurrentmin = thenewpoint
    return thecurrentmax, thecurrentmin


def optical2spo2(irdata, visdata, samplerate, windowwidth=1.0, verbose=False):
    if verbose:
        print("[optical2spo2]")
    irlow = dolptrapfftfilt(samplerate, 0.45, 0.5, irdata)
    irhigh = irdata - irlow
    if verbose:
        print("[optical2spo2] done filtering ir")
    vislow = dolptrapfftfilt(samplerate, 0.45, 0.5, visdata)
    vishigh = visdata - vislow
    if verbose:
        print("[optical2spo2] done filtering ir & vs")

    winsize = int(windowwidth * samplerate)
    irmax = np.zeros((len(irhigh)), dtype="float")
    irmin = np.zeros((len(irhigh)), dtype="float")
    vismax = np.zeros((len(vishigh)), dtype="float")
    vismin = np.zeros((len(vishigh)), dtype="float")

    irmax[: int(winsize / 2)] = np.max(irhigh[: int(winsize / 2)])
    irmin[: int(winsize / 2)] = np.min(irhigh[: int(winsize / 2)])
    vismax[: int(winsize / 2)] = np.max(vishigh[: int(winsize / 2)])
    vismin[: int(winsize / 2)] = np.min(vishigh[: int(winsize / 2)])
    for i in range(int(winsize / 2), len(irhigh) - int(winsize / 2)):
        irmax[i] = np.max(irhigh[i - int(winsize / 2) : i + int(winsize / 2)])
        irmin[i] = np.min(irhigh[i - int(winsize / 2) : i + int(winsize / 2)])
        vismax[i] = np.max(vishigh[i - int(winsize / 2) : i + int(winsize / 2)])
        vismin[i] = np.min(vishigh[i - int(winsize / 2) : i + int(winsize / 2)])
    irmax[-int(winsize / 2) :] = np.max(irhigh[-int(winsize / 2) :])
    irmin[-int(winsize / 2) :] = np.min(irhigh[-int(winsize / 2) :])
    vismax[-int(winsize / 2) :] = np.max(vishigh[-int(winsize / 2) :])
    vismin[-int(winsize / 2) :] = np.min(vishigh[-int(winsize / 2) :])
    irac = irmax - irmin
    visac = vismax - vismin
    return dolptrapfftfilt(
        samplerate, 0.25, 0.3, 110.0 - 25.0 * (visac / vislow) / (irac / irlow) - 2.5
    )


def rt_smoothGSR(therawdata, thetimepoint, thesmootheddata, fps, reinitsmoothFIR=False):
    global smoothFIRinited, smoothfiltcoffs
    global smoothFIRlen

    if (not smoothFIRinited) or reinitsmoothFIR:
        smoothFIRlen = 61
        print(
            "initializing FIR filters for ",
            fps,
            "Hz sample rate, ",
            smoothFIRlen / fps,
            ", second prefill",
        )
        smoothfiltcoffs = design_LFO_filter(fps, smoothFIRlen)
        smoothFIRinited = True
    smoothfiltlen = len(smoothfiltcoffs)
    if thetimepoint >= smoothfiltlen:
        thesmootheddata[thetimepoint] = np.dot(
            therawdata[thetimepoint - smoothfiltlen : thetimepoint], smoothfiltcoffs
        )


def rt_QRSdiscriminator(
    therawdata, thetimepoint, theqrsdata, actualsamplerate, reinitQRSFIR=False
):
    global QRSFIRinited, QRSbpfiltcoffs
    global QRSFIRlen
    if (not QRSFIRinited) or reinitQRSFIR:
        QRSFIRlen = int(
            round(QRSFIRtime * actualsamplerate)
        )  # convert from time to points
        QRSFIRlen += 1 - (QRSFIRlen % 2)  # make sure QRSFIRlen is odd
        QRSbpfiltcoffs = design_QRS_filter(actualsamplerate, QRSFIRlen)
        print("inited QRS higpass filter")
        # print('filtercoffs:')
        # print(QRSbpfiltcoffs)
        QRSFIRinited = True
    QRSbpfiltlen = len(QRSbpfiltcoffs)
    if thetimepoint >= QRSbpfiltlen:
        theqrsdata[thetimepoint] = np.abs(
            np.dot(
                therawdata[thetimepoint - QRSbpfiltlen : thetimepoint], QRSbpfiltcoffs
            )
        )


# """ Uses physioFIRlen last values ([timepoint - 61:time point]) from card data arrays; in this case, 61 """
# def rt_splitdata(therawdata, thetimepoint, thecarddata, thecardsq, thecardrms, therespdata, thelfodata, fps,
#                  reinitphysioFIR=False):
#
#     global physioFIRinited, cardbpfiltcoffs, respbpfiltcoffs, lfolpfiltcoffs, cardsmoothfiltcoffs
#     global physioFIRlen
#
#     if (not physioFIRinited) or reinitphysioFIR:
#         physioFIRlen = int(round(physioFIRtime * fps))   # convert from time to points
#         physioFIRlen += (1 - (physioFIRlen % 2))         # make sure physioFIRlen is odd
#         print('initializing FIR filters for ', fps, 'Hz sample rate, ', physioFIRlen / fps, ', second prefill')
#         cardbpfiltcoffs = design_card_filter(fps, physioFIRlen)
#         respbpfiltcoffs = design_resp_filter(fps, physioFIRlen)
#         lfolpfiltcoffs = design_LFO_filter(fps, physioFIRlen)
#         cardsmoothfiltcoffs = design_cardsmooth_filter(fps, physioFIRlen)
#         physioFIRinited = True
#
#     cardbpfiltlen = len(cardbpfiltcoffs)
#     respbpfiltlen = len(respbpfiltcoffs)
#     lfolpfiltlen = len(lfolpfiltcoffs)
#     cardsmoothfiltlen = len(cardsmoothfiltcoffs)
#
#     if thetimepoint >= cardbpfiltlen:
#         thecarddata[thetimepoint] = np.dot(therawdata[thetimepoint - cardbpfiltlen:thetimepoint], cardbpfiltcoffs)
#     if thetimepoint >= respbpfiltlen:
#         therespdata[thetimepoint] = np.dot(therawdata[thetimepoint - respbpfiltlen:thetimepoint], respbpfiltcoffs)
#     if thetimepoint >= lfolpfiltlen:
#         thelfodata[thetimepoint] = np.dot(therawdata[thetimepoint - lfolpfiltlen:thetimepoint], lfolpfiltcoffs)
#     thecardsq[thetimepoint] = thecarddata[thetimepoint] * thecarddata[thetimepoint]
#     if thetimepoint >= cardsmoothfiltlen:
#         thecardrms[thetimepoint] = np.sqrt(
#             np.dot(thecardsq[thetimepoint - cardsmoothfiltlen:thetimepoint], cardsmoothfiltcoffs))

""" Uses physioFIRlen last values ([timepoint - 61:time point]) from card data arrays; in this case, 61 """
""" Changed this to deal with scalars instead of arrays for measures where recent history is not needed. As
we are dealing with immutable types, modified code to return the values instead of setting array vals in the method.
"""
# def rt_splitdata(therawdata, thetimepoint, thecarddatapt, thecardsq, thecardrmspt, therespdatapt, thelfodatapt, fps,
#                  reinitphysioFIR=False):
"""
thecardsq: CircularList that does not contain anything for thetimepoint yet.
therawdata: CircularList

A lot is based off assumption that filter lengths are all <= maxdelay. All circular list lengths
are determined under that assumption.
"""


# NOA - have now specced physioFIRtime rather than len
def rt_splitdata(therawdata, thetimepoint, thecardsq, fps, reinitphysioFIR=False):
    global physioFIRinited, cardbpfiltcoffs, respbpfiltcoffs, lfolpfiltcoffs, cardsmoothfiltcoffs
    global physioFIRlen

    if (not physioFIRinited) or reinitphysioFIR:
        physioFIRlen = int(round(physioFIRtime * fps))  # convert from time to points
        physioFIRlen += 1 - (physioFIRlen % 2)  # make sure physioFIRlen is odd
        print(
            "initializing FIR filters for ",
            fps,
            "Hz sample rate, ",
            physioFIRlen / fps,
            ", second prefill",
        )
        cardbpfiltcoffs = design_card_filter(fps, physioFIRlen)
        respbpfiltcoffs = design_resp_filter(fps, physioFIRlen)
        lfolpfiltcoffs = design_LFO_filter(fps, physioFIRlen)
        cardsmoothfiltcoffs = design_cardsmooth_filter(fps, physioFIRlen)
        physioFIRinited = True

    cardbpfiltlen = len(cardbpfiltcoffs)
    respbpfiltlen = len(respbpfiltcoffs)
    lfolpfiltlen = len(lfolpfiltcoffs)
    cardsmoothfiltlen = len(cardsmoothfiltcoffs)

    # Initializing to nan.
    # TODO is that the value they should have while we're waiting to reach the filter length?
    thecarddatapt, thecardrmspt, therespdatapt, thelfodatapt = (
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    )

    if thetimepoint >= cardbpfiltlen:
        thecarddatapt = np.dot(therawdata.get_last(cardbpfiltlen), cardbpfiltcoffs)
    if thetimepoint >= respbpfiltlen:
        therespdatapt = np.dot(therawdata.get_last(respbpfiltlen), respbpfiltcoffs)
    if thetimepoint >= lfolpfiltlen:
        thelfodatapt = np.dot(therawdata.get_last(lfolpfiltlen), lfolpfiltcoffs)

    # add info for thetimepoint to thecardsq, a circular list
    thecardsq.append(thecarddatapt * thecarddatapt)
    # thecardsq[thetimepoint] = thecarddatapt * thecarddatapt
    if thetimepoint >= cardsmoothfiltlen:
        # TODO what should thecardrms be while timepoint < cardsmoothfiltlen?? nan??
        thecardrmspt = np.sqrt(
            np.dot(thecardsq.get_last(cardsmoothfiltlen), cardsmoothfiltcoffs)
        )
        # thecardrmspt = np.sqrt(np.dot(thecardsq[thetimepoint - cardsmoothfiltlen:thetimepoint], cardsmoothfiltcoffs))

    return thecarddatapt, thecardrmspt, therespdatapt, thelfodatapt


def rt_splitacdcdata(
    therawdata, thetimepoint, theDCdata, theACdata, fps, reinitacdcFIR=False
):
    global acdcFIRinited, dcfiltcoffs
    global acdcFIRlen

    if (not acdcFIRinited) or reinitacdcFIR:
        acdcFIRlen = int(round(acdcFIRtime * fps))  # convert from time to points
        acdcFIRlen += 1 - (acdcFIRlen % 2)  # make sure acdcFIRlen is odd
        print(
            "initializing FIR filters for ",
            fps,
            "Hz sample rate, ",
            acdcFIRlen / fps,
            ", second prefill",
        )
        dcfiltcoffs = design_DC_filter(fps, acdcFIRlen)
        acdcFIRinited = True

    if thetimepoint >= acdcFIRlen:
        theDCdata[thetimepoint] = np.dot(
            therawdata[thetimepoint - acdcFIRlen : thetimepoint], dcfiltcoffs
        )
        theACdata[thetimepoint] = therawdata[thetimepoint] - theDCdata[thetimepoint]


# def updateheartrate(samplerate, carddata, cardrms, thetimepoint, currentheartrate, cardpeak):
#     if (carddata[thetimepoint] / cardrms[thetimepoint] > hbthresh):
#         peakregion = 1
#     else:
#         peakregion = 0
#     # print('\t\tpeakregion: ', peakregion, 'carddata:', carddata[thetimepoint], 'cardrms:', cardrms[thetimepoint])
#     if peakregion:
#         if (carddata[thetimepoint] < carddata[thetimepoint - 1]) and (thetimepoint - cardpeak[-1] > 9):
#             cardpeak.append(thetimepoint)
#             print("\t\tcardpeak at ",thetimepoint)
#             pointsdiff = (cardpeak[-1] - cardpeak[-(numhbstoavg + 1)])
#             timediff = pointsdiff / samplerate
#             currentheartrate = numhbstoavg * 60.0 / timediff
#     return currentheartrate


# def rt_updateheartrate(samplerate, carddata, cardrmspt, thetimepoint, currentheartrate, cardpeak):
#     if (carddata[thetimepoint] / cardrmspt > hbthresh):
#         peakregion = 1
#     else:
#         peakregion = 0
#     # print('\t\tpeakregion: ', peakregion, 'carddata:', carddata[thetimepoint], 'cardrms:', cardrms[thetimepoint])
#     if peakregion:
#         if (carddata[thetimepoint] < carddata[thetimepoint - 1]) and (thetimepoint - cardpeak[-1] > 9):
#             cardpeak.append(thetimepoint)
#             print("\t\tcardpeak at ",thetimepoint)
#             pointsdiff = (cardpeak[-1] - cardpeak[-(numhbstoavg + 1)])
#             timediff = pointsdiff / samplerate
#             currentheartrate = numhbstoavg * 60.0 / timediff
#     return currentheartrate

# need carddata [ -2]
# need cardpeak [-(numhbstoavg + 1)]
"""
samplerate
carddata -- circularlist
cardpeak -- circularlist
"""


def rt_updateheartrate(
    samplerate, carddata, cardrmspt, thetimepoint, currentheartrate, cardpeak
):
    has_peak = 0
    if carddata[-1] / cardrmspt > hbthresh:
        peakregion = 1
    else:
        peakregion = 0
    if peakregion:
        if (carddata[-1] < carddata[-2]) and (thetimepoint - cardpeak[-1] > 9):
            has_peak = 1
            cardpeak.append(thetimepoint)
            print("\t\tcardpeak at ", thetimepoint)
            pointsdiff = cardpeak[-1] - cardpeak[-(numhbstoavg + 1)]
            timediff = pointsdiff / samplerate
            currentheartrate = numhbstoavg * 60.0 / timediff
    return currentheartrate, has_peak


# perform MARA baseline correction on a signal
def MARA(
    insignal,
    samplerate,
    wintime=5.0,
    threshhold=-1.0,
    pvalue=0.05,
    padtime=10.0,
    debug=False,
):
    # step 0 - pad if necessary
    padlen = int(padtime * samplerate)
    if padlen > 0:
        signal = padvec(insignal, padlen=padlen)
    else:
        signal = insignal

    # step 1 - calculate the two sided moving averag of the signal
    themsd = msd(signal, N=int((wintime / 2.0) * samplerate), debug=debug)

    # step 2 - determine the threshhold (if not specified
    if threshhold < 0.0:
        medianval = np.median(themsd)
        sigma = mad(themsd, center=medianval)
        threshhold = medianval + np.sqrt(2.0) * erfinv(1.0 - pvalue) * sigma
        if debug:
            print("median, sigma, threshhold of", medianval, sigma, threshhold)

    # step 3 - find the motion artifact segments
    thesegmentlist = detectMA(themsd, threshhold, debug=False)

    # step 4 - generate the spline fits to the artifact baselines
    timestep = 1.0 / samplerate
    tvals = np.arange(0.0, len(signal) * timestep, timestep)
    thesplines = getsplines(tvals, signal, thesegmentlist)

    # step 5 - subtract off splines
    desplined = 1.0 * signal
    for idx, segment in enumerate(thesegmentlist):
        if segment[2]:
            startpt = thesegmentlist[idx][0]
            endpt = thesegmentlist[idx][1]
            desplined[startpt : endpt + 1] -= thesplines[idx]

    # step 6 - line up segment ends
    alpha = int(0.3333 * samplerate)
    beta = int(2.0 * samplerate)
    aligned = 1.0 * desplined
    for idx in range(1, len(thesegmentlist[1:])):
        segment1 = aligned[thesegmentlist[idx - 1][0] : thesegmentlist[idx - 1][1] + 1]
        segment2 = aligned[thesegmentlist[idx][0] : thesegmentlist[idx][1] + 1]
        lambda1 = len(segment1)
        theta1 = int(lambda1 // 10)
        lambda2 = len(segment2)
        theta2 = int(lambda2 // 10)
        if lambda1 <= alpha:
            if lambda2 <= alpha:
                a = np.mean(segment1)
                b = np.mean(segment2)
            elif lambda2 >= beta:
                a = np.mean(segment1)
                b = np.mean(segment2[: theta2 - 1])
            else:
                a = np.mean(segment1)
                b = np.mean(segment2[: alpha - 1])
        elif lambda1 >= beta:
            if lambda2 <= alpha:
                a = np.mean(segment1[lambda1 - theta1 - 1 :])
                b = np.mean(segment2)
            elif lambda2 >= beta:
                a = np.mean(segment1[lambda1 - theta1 - 1 :])
                b = np.mean(segment2[: theta2 - 1])
            else:
                a = np.mean(segment1[lambda1 - theta1 - 1 :])
                b = np.mean(segment2[: alpha - 1])
        else:
            if lambda2 <= alpha:
                a = np.mean(segment1[lambda1 - alpha - 1 :])
                b = np.mean(segment2)
            elif lambda2 >= beta:
                a = np.mean(segment1[lambda1 - alpha - 1 :])
                b = np.mean(segment2[: theta2 - 1])
            else:
                a = np.mean(segment1[lambda1 - alpha - 1 :])
                b = np.mean(segment2[: alpha - 1])

        thissegstart = thesegmentlist[idx][0]
        thissegend = thesegmentlist[idx][1] + 1
        aligned[thissegstart:thissegend] = aligned[thissegstart:thissegend] + a - b

    if debug:
        plt.plot(signal)
        plt.plot(desplined)
        plt.plot(aligned)
        plt.legend(["signal", "desplined", "aligned"])
        plt.show()

    if padlen > 0:
        return unpadvec(aligned, padlen=padlen)
    else:
        return aligned


# calculate the moving standard deviation of a signal over a length N
def msd(signal, N=250, debug=False):
    if debug:
        print("using a window of", 2 * N + 1, "points")
    themsd = 0.0 * signal
    for i in range(1, len(signal)):
        themsd[i] = np.std(
            signal[np.max((0, i - N)) : np.min((len(signal) - 1, i + N))]
        )
    if debug:
        plt.plot(signal)
        plt.plot(themsd)
        plt.legend(["signal", "themsd"])
        plt.show()
    return themsd


def detectMA(themsd, threshhold, debug=False):
    mask = np.where(themsd >= threshhold, 1.0, 0.0)
    if debug:
        plt.plot(mask)
        plt.legend(["mask"])
        plt.show()

    # find artifact regions where msd exceeds threshhold and segment is at least 5 points
    segmentlist = []
    location = 0
    isMA = False
    currentstart = 0
    while location < len(themsd):
        if not isMA:
            if mask[location] > 0.5:
                if location - currentstart > 5:
                    segmentlist.append([currentstart, location - 1, isMA])
                    currentstart = location
                    isMA = True
        else:
            if mask[location] < 0.5:
                if location - currentstart > 5:
                    segmentlist.append([currentstart, location - 1, isMA])
                    currentstart = location
                    isMA = False
        location += 1

    if location - currentstart > 5:
        segmentlist.append([currentstart, location - 1, isMA])
    else:
        segmentlist[-1] = [segmentlist[-1][0], location - 1, segmentlist[-1][2]]

    if debug:
        print("the full segment list")
        for theelement in segmentlist:
            print(theelement)
    return segmentlist


def getsplines(tvals, signal, segmentlist):
    thesplines = []
    for theregion in segmentlist:
        if theregion[2]:
            m = 1.0 * (theregion[1] - theregion[0] + 1)
            thesval = (m + np.sqrt(2.0 * m)) * np.square(
                np.std(signal[theregion[0] : theregion[1]] + 1)
            )
            tck = interpolate.splrep(
                tvals[theregion[0] : theregion[1] + 1],
                signal[theregion[0] : theregion[1] + 1],
                s=thesval,
            )
            thesplines.append(
                interpolate.splev(tvals[theregion[0] : theregion[1] + 1], tck)
            )
        else:
            thesplines.append(None)
    return thesplines


# applies the wavelet transform to the signal, then thresholds
# the coefficients to reduce noise
# based on Molavi et al, doi: 10.1117/12.875741
def wavelet_despike(
    data, pvalue=0.05, thewavelet="db6", verbose=False, avoid=None, debug=False
):
    if verbose:
        print("applying forward wavelet transform...")

    if debug:
        print(f"wavelet_despike: data length in is {len(data)}")

    # apply the wavelet tranform to each column (fNIRS channel)
    coeffs = pywt.wavedec(data, thewavelet, mode="periodic")
    if debug:
        for j in range(len(coeffs)):
            with open("coeffs_" + str(j).zfill(2), "w") as FILE:
                for i in coeffs[j]:
                    FILE.writelines(str(i) + "\n")

    # define the soft threshold for the coefficients
    uthresh = []
    med = []
    sigma = []
    scalesize = []
    for i in range(0, len(coeffs)):
        scalesize.append(len(data) / len(coeffs[i]))
    for i in range(1, len(coeffs)):
        if verbose:
            print(len(coeffs[i]), "coefficients at level", i)
        med.append(np.median(coeffs[i]))
        sigma.append(mad(coeffs[i], center=med[-1]))
        uthresh.append(np.sqrt(2.0) * erfinv(1.0 - pvalue) * sigma[-1])
        if verbose:
            print(
                "at scale",
                i,
                "median=",
                med[-1],
                "sigma=",
                sigma[-1],
                "uthresh=",
                uthresh[-1],
            )
    if debug:
        with open("median", "w") as FILE:
            for i in med:
                FILE.writelines(str(i) + "\n")
        with open("sigma", "w") as FILE:
            for i in sigma:
                FILE.writelines(str(i) + "\n")
        with open("scalesize", "w") as FILE:
            for i in scalesize:
                FILE.writelines(str(i) + "\n")

    # apply the threshold and reconstruct the signal
    denoised = coeffs[:]
    numtoremove = []
    for j in range(1, len(denoised)):
        numtoremove.append(np.sum(np.where(abs(coeffs[j]) > uthresh[j - 1], 1, 0)))
        if verbose:
            print(
                "at level",
                j,
                ",",
                numtoremove[-1],
                "of",
                len(coeffs[j]),
                "(",
                100.0 * numtoremove[-1] / len(coeffs[j]),
                "%) are out of bounds",
            )
    for j in range(1, len(denoised)):
        if verbose:
            print("cleaning level", j)
        skip = False
        if avoid is not None:
            if (avoid / 2.0) <= scalesize[j] <= (2.0 * avoid):
                print("skipping filtration on scale", scalesize[j])
                skip = True
        if not skip:
            denoised[j] = np.where(
                abs(coeffs[j] - med[j - 1]) > uthresh[j - 1], med[j - 1], coeffs[j]
            )
    if debug:
        for j in range(len(coeffs)):
            with open("denoised_" + str(j).zfill(2), "w") as FILE:
                for i in denoised[j]:
                    FILE.writelines(str(i) + "\n")
    new_data = pywt.waverec(denoised, thewavelet, mode="periodic")[0 : len(data)]

    if debug:
        print(f"wavelet_despike: data length out is {len(new_data)}")

    if verbose:
        print("done")
    return new_data


def multidespike(
    data,
    nsegs=2,
    pvalue=0.05,
    thewavelet="db10",
    avoid=None,
    verbose=False,
    debug=False,
):
    outdata = 0.0 * data
    for i in range(0, nsegs):
        segshift = i * int(len(data) / nsegs)
        if verbose:
            print("shifting", segshift, "for segment", i)
        outdata += np.roll(
            wavelet_despike(
                np.roll(data, segshift),
                pvalue=pvalue,
                thewavelet=thewavelet,
                avoid=avoid,
                verbose=verbose,
                debug=debug,
            ),
            -segshift,
        )
    return outdata / nsegs


def despike(invec):
    themean = np.mean(invec)
    pspikes = np.where(invec > 2 * themean)[0]
    for i in range(0, len(pspikes)):
        if pspikes[i] == 0:
            invec[pspikes[i]] = invec[pspikes[i] + 1]
        elif pspikes[i] == len(invec) - 1:
            invec[pspikes[i]] = invec[pspikes[i] - 1]
        else:
            invec[pspikes[i]] = (invec[pspikes[i] - 1] + invec[pspikes[i] + 1]) / 2.0
    nspikes = np.where(invec < 0.5 * themean)[0]
    for i in range(0, len(nspikes)):
        if nspikes[i] == 0:
            invec[nspikes[i]] = invec[nspikes[i] + 1]
        elif nspikes[i] == len(invec) - 1:
            invec[nspikes[i]] = invec[nspikes[i] - 1]
        else:
            invec[nspikes[i]] = (invec[nspikes[i] - 1] + invec[nspikes[i] + 1]) / 2.0
    return invec


# this is the readvecs function
def readvecs(inputfilename):
    file = open(inputfilename)
    lines = file.readlines()
    numvecs = len(lines[0].split())
    inputvec = np.zeros((numvecs, MAXLINES), dtype="float")

    numvals = 0
    for line in lines:
        numvals += 1
        thetokens = line.split()
        for vecnum in range(0, numvecs):
            inputvec[vecnum, numvals - 1] = float(thetokens[vecnum])
    return 1.0 * inputvec[:, 0:numvals]


def hamming(length):
    return 0.54 - 0.46 * np.cos(
        np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / float(length))
    )


# http://stackoverflow.com/questions/12323959/fast-cross-correlation-method-in-python
def fastcorrelate(input1, input2, usefft=True, mode="full"):
    if usefft:
        # Do an array flipped convolution, which is a correlation.
        return np.fftconvolve(input1, input2[::-1], mode=mode)
    else:
        return np.correlate(input1, input2, mode=mode)


def normxcorr(vector1, vector2, Fs, thelabel):
    vec1len = len(vector1)
    vec2len = len(vector2)
    # print(vec1len,vec2len)
    thexcorr = fastcorrelate(
        hamming(vec1len) * vector1, hamming(vec2len) * vector2, mode="full"
    )
    xcorrlen = len(thexcorr)
    # print(vec1len,vec2len,xcorrlen)
    normfac1 = fastcorrelate(
        hamming(vec1len) * vector1, hamming(vec1len) * vector1, mode="full"
    )[int(xcorrlen / 2)]
    normfac2 = fastcorrelate(
        hamming(vec2len) * vector2, hamming(vec2len) * vector2, mode="full"
    )[int(xcorrlen / 2)]
    thenormxcorr = thexcorr / np.sqrt(normfac1 * normfac2)
    sampletime = 1.0 / Fs
    xcorr_x = np.r_[0.0:xcorrlen] * sampletime - ((xcorrlen - 1) * sampletime) / 2.0
    halfwindow = int(searchrange * Fs)
    searchstart = int(xcorrlen) / 2 - halfwindow
    searchend = int(xcorrlen) / 2 + halfwindow
    xcorr_x_trim = xcorr_x[searchstart:searchend]
    thenormxcorr_trim = thenormxcorr[searchstart:searchend]
    thedelay = xcorr_x_trim[np.argmax(thenormxcorr_trim)]
    # print(thelabel, "maxdelay=",-thedelay, " seconds")
    return thedelay, xcorr_x, thenormxcorr


def doresample(orig_x, orig_y, new_x):
    cj = cspline1d(orig_y)
    return cspline1d_eval(cj, new_x, dx=(orig_x[1] - orig_x[0]), x0=orig_x[0])


def hrfft(samplefreq, indata, hrfftsize, timepoint, verbose=True):
    thefreqs = (samplefreq / hrfftsize) * np.arange(0, int(hrfftsize // 2))
    thedata = hamming(hrfftsize) * indata[timepoint - hrfftsize : timepoint]
    thefftdata = abs(np.fft.fft(thedata))
    thefftheartrate = thefreqs[np.argmax(thefftdata[0 : int(hrfftsize // 2)])]
    if verbose:
        return thefreqs, thefftdata[0 : int(hrfftsize / 2)], thefftheartrate
    else:
        return thefftheartrate


"""
TODO output of this seems off.
"""


def rt_hrfft(samplefreq, indata, hrfftsize, timepoint, verbose=True):
    """
    indata is circular list with size hrfftsize

    """
    thefreqs = (samplefreq / hrfftsize) * np.arange(0, int(hrfftsize // 2))
    thedata = hamming(hrfftsize) * indata.get_last(hrfftsize)
    thefftdata = abs(np.fft.fft(thedata))
    thefftheartrate = thefreqs[np.argmax(thefftdata[0 : int(hrfftsize // 2)])]
    return thefreqs, thefftdata[0 : int(hrfftsize / 2)], thefftheartrate
    # a boolean parameter shouldn't switch up number of returned args...
    # if verbose:
    #     return thefreqs, thefftdata[0:int(hrfftsize / 2)], thefftheartrate
    # else:
    #     return thefftheartrate


def vechrfft(samplefreq, indata, hrffttime=8.5):
    theffthr = indata * 0.0
    hrfftsize = int(hrffttime * samplefreq)
    for i in range(hrfftsize, len(indata)):
        theffthr[i] = 60.0 * hrfft(samplefreq, indata, hrfftsize, i, verbose=False)
    theffthr[0 : hrfftsize - 1] = theffthr[hrfftsize]
    return theffthr


def arb_pass(samplerate, inputdata, usebutterworthfilter, arb_lowerpass, arb_upperpass):
    if usebutterworthfilter:
        return tide_filt.dohpfiltfilt(
            samplerate,
            arb_lowerpass,
            tide_filt.dolpfiltfilt(
                samplerate, arb_upperpass, inputdata, defaultbutterorder
            ),
            2,
        )
    else:
        if trapezoidalfftfilter:
            return dobptrapfftfilt(
                samplerate,
                0.9 * arb_lowerpass,
                arb_lowerpass,
                arb_upperpass,
                1.1 * arb_upperpass,
                inputdata,
            )
        else:
            return tide_filt.dobpfftfilt(
                samplerate, arb_lowerpass, arb_upperpass, inputdata
            )


def lfo_pass(samplerate, inputdata, usebutterworthfilter):
    if usebutterworthfilter:
        return tide_filt.dohpfiltfilt(
            samplerate,
            lf_lowerpass,
            tide_filt.dolpfiltfilt(
                samplerate, lf_upperpass, inputdata, defaultbutterorder
            ),
            2,
        )
    else:
        if trapezoidalfftfilter:
            return tide_filt.dobptrapfftfilt(
                samplerate,
                lf_lowerstop,
                lf_lowerpass,
                lf_upperpass,
                lf_upperstop,
                inputdata,
            )
        else:
            return tide_filt.dobpfftfilt(
                samplerate, lf_lowerpass, lf_upperpass, inputdata
            )


def resp_pass(samplerate, inputdata, usebutterworthfilter):
    if usebutterworthfilter:
        return tide_filt.dobpfiltfilt(
            samplerate, resp_lowerpass, resp_upperpass, inputdata, defaultbutterorder
        )
    else:
        if trapezoidalfftfilter:
            return tide_filt.dobptrapfftfilt(
                samplerate,
                resp_lowerstop,
                resp_lowerpass,
                resp_upperpass,
                resp_upperstop,
                inputdata,
            )
        else:
            return tide_filt.dobpfftfilt(
                samplerate, resp_lowerpass, resp_upperpass, inputdata
            )


def card_pass(samplerate, inputdata, usebutterworthfilter):
    if usebutterworthfilter:
        return tide_filt.dobpfiltfilt(
            samplerate, card_lowerpass, card_upperpass, inputdata, defaultbutterorder
        )
    else:
        if trapezoidalfftfilter:
            return tide_filt.dobptrapfftfilt(
                samplerate,
                card_lowerstop,
                card_lowerpass,
                card_upperpass,
                card_upperstop,
                inputdata,
            )
        else:
            return tide_filt.dobpfftfilt(
                samplerate, card_lowerpass, card_upperpass, inputdata
            )


def padvec(indata, padlen=1000):
    return np.concatenate((indata[::-1][-padlen:], indata, indata[::-1][0:padlen]))


def unpadvec(indata, padlen=1000):
    return indata[padlen:-padlen]


def dolptrapfftfilt(Fs, Fpass, Fstop, indata, padlen=100):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = np.fft.fft(padindata)
    filterfunc = getlptrapfftfunc(Fs, Fpass, Fstop, padindata)
    indata_trans = indata_trans * filterfunc
    return unpadvec(np.fft.ifft(indata_trans).real, padlen=padlen)


def dobptrapfftfilt(Fs, Fstopl, Fpassl, Fpassu, Fstopu, indata, padlen=100):
    padindata = padvec(indata, padlen=padlen)
    indata_trans = np.fft.fft(padindata)
    filterfunc = getlptrapfftfunc(Fs, Fpassu, Fstopu, padindata) * (
        1.0 - getlptrapfftfunc(Fs, Fstopl, Fpassl, padindata)
    )
    indata_trans = indata_trans * filterfunc
    return unpadvec(np.fft.ifft(indata_trans).real, padlen=padlen)


def getlptrapfftfunc(Fs, Fpass, Fstop, indata):
    filterfunc = np.ones(len(indata), dtype="float")
    passbin = int((Fpass / Fs) * len(filterfunc))
    cutoffbin = int((Fstop / Fs) * len(filterfunc))
    translength = cutoffbin - passbin
    transvector = np.arange(1.0 * translength) / translength
    if translength > 0:
        filterfunc[passbin:cutoffbin] = 1.0 - transvector
    if cutoffbin != len(indata):
        filterfunc[cutoffbin:-cutoffbin] = 0.0
    if translength > 0:
        filterfunc[-cutoffbin:-passbin] = transvector
    return filterfunc


class noncausalprefilter:
    def __init__(self, type="none", usebutterworth=False, butterworthorder=3):
        self.filtertype = type
        self.arb_lower = 0.05
        self.arb_upper = 0.20
        self.usebutterworth = usebutterworth
        self.butterworthorder = butterworthorder

    def settype(self, type):
        self.filtertype = type

    def gettype(self):
        return self.filtertype

    def setbutter(self, useit, order=3):
        self.usebutterworth = useit
        self.butterworthorder = order

    def setarb(self, lower, upper):
        self.arb_lower = 1.0 * lower
        self.arb_upper = 1.0 * upper

    def apply(self, samplerate, data):
        if self.filtertype == "lfo":
            return lfo_pass(samplerate, data, self.usebutterworth).real
        elif self.filtertype == "resp":
            return resp_pass(samplerate, data, self.usebutterworth).real
        elif self.filtertype == "cardiac":
            return card_pass(samplerate, data, self.usebutterworth).real
        elif self.filtertype == "cardsmooth":
            return env_pass(
                samplerate, data, self.usebutterworth, env_upperpass=0.2
            ).real
        elif self.filtertype == "arb":
            return arb_pass(
                samplerate, data, self.usebutterworth, self.arb_lower, self.arb_upper
            ).real
        elif self.filtertype == "none":
            return data
        else:
            print("bad prefilter type")
            exit()


def gauss_eval(x, p):
    return p[0] * np.exp(-((x - p[1]) ** 2) / (2 * p[2] ** 2))


def gaussresiduals(p, y, x):
    err = y - gauss_eval(x, p)
    return err


def gaussfit(height, loc, width, xvals, yvals):
    plsq, dummy = leastsq(
        gaussresiduals, np.array([height, loc, width]), args=(yvals, xvals), maxfev=5000
    )
    return (plsq[0], plsq[1], plsq[2])


def env_pass(samplerate, inputdata, usebutterworthfilter, env_upperpass=0.5):
    if usebutterworthfilter:
        return tide_filt.dolpfiltfilt(
            samplerate, env_upperpass, inputdata, defaultbutterorder
        )
    else:
        if trapezoidalfftfilter:
            return tide_filt.dolptrapfftfilt(
                samplerate, env_upperpass, 1.1 * env_upperpass, inputdata
            )
        else:
            return tide_filt.dolpfftfilt(samplerate, env_upperpass, inputdata)


def normalize_cardiac(
    data, samplerate, env_upperpass=0.5, thresh=0.3, winsizeinsecs=1.0
):
    winsizeinpoints = int(winsizeinsecs * samplerate)

    localmax = np.zeros((len(data)), dtype="float")
    localmin = np.zeros((len(data)), dtype="float")
    localrange = np.zeros((len(data)), dtype="float")
    bcdata = np.zeros((len(data)), dtype="float")
    bcnormdata = np.zeros((len(data)), dtype="float")
    envnormdata = np.zeros((len(data)), dtype="float")
    valid = np.ones((len(data)), dtype="float")

    # do a local normalization
    print(len(data))
    envelope = env_pass(samplerate, np.abs(data), False, env_upperpass=env_upperpass)
    for i in range(1, len(data)):
        localmax[i] = np.max(data[max([0, i - winsizeinpoints]) : i])
        localmin[i] = np.min(data[max([0, i - winsizeinpoints]) : i])
        localrange[i] = localmax[i] - localmin[i]
        bcdata[i] = data[i] - localmin[i]
        if localrange[i] > 0.0:
            bcnormdata[i] = bcdata[i] / localrange[i]
            envnormdata[i] = data[i] / (2.0 * envelope[i])

    # do some rudimentary filtering
    rangemed = np.median(localrange)
    for i in range(1, len(data)):
        if localrange[i] < thresh * rangemed or localrange[i] > 3.0 * rangemed:
            bcdata[i] = 0.0
            bcnormdata[i] = 0.0
            envnormdata[i] = 0.0
            valid[i] = 0.0

    return rangemed, envelope, envnormdata, valid


def procpeaks(
    timeaxis,
    waveform,
    minlocs,
    maxlocs,
    samplerate,
    usemin=False,
    peakfilter=True,
    debug=False,
):
    print("processing peaks")
    peakdictlist = []
    rrinterval = np.zeros((len(timeaxis)), dtype="float")
    peakmarkers = np.zeros((len(timeaxis)), dtype="float")
    peakheight = np.zeros((len(timeaxis)), dtype="float")
    confidence = np.zeros((len(timeaxis)), dtype="float")
    instbpm = np.zeros((len(timeaxis)), dtype="float")
    avgbpm = np.zeros((len(timeaxis)), dtype="float")

    # coalesce peak information - a "peak" goes from minimum to maximum
    for i in range(0, len(minlocs) - 1):
        peakdictlist.append({})
        peakdictlist[-1]["starttime"] = 1.0 * minlocs[i, 0]
        peakdictlist[-1]["endtime"] = 1.0 * minlocs[i + 1, 0]
        peakdictlist[-1]["risetime"] = maxlocs[i, 0] - minlocs[i, 0]
        peakdictlist[-1]["falltime"] = minlocs[i + 1, 0] - maxlocs[i, 0]
        peakdictlist[-1]["height"] = 1.0 * maxlocs[i, 1] - minlocs[i, 1]
        peakdictlist[-1]["bldiff"] = 1.0 * minlocs[i + 1, 1] - minlocs[i, 1]
        peakdictlist[-1]["confidence"] = 0.0
    for i in range(1, len(minlocs) - 1):
        peakdictlist[i]["rri"] = (
            peakdictlist[i]["starttime"] - peakdictlist[i - 1]["starttime"]
        )
    if len(peakdictlist) > 1:
        peakdictlist[0]["rri"] = peakdictlist[1]["rri"]

    # do the actual filtering here
    if peakfilter:
        filteredpeakdictlist = filterpeaklist(peakdictlist, debug=debug)
    else:
        filteredpeakdictlist = peakdictlist

    # calculate summary statistics
    rrilist = []
    risetimelist = []
    falltimelist = []
    heightlist = []

    for index, thepeak in enumerate(filteredpeakdictlist):
        if thepeak["confidence"] == 1.0 or (not peakfilter):
            rrilist.append(thepeak["rri"])
            risetimelist.append(thepeak["risetime"])
            falltimelist.append(thepeak["falltime"])
            heightlist.append(thepeak["height"])
    print("Summary statistics:")
    print(
        "   RRI =      ",
        np.mean(rrilist),
        "+/-",
        np.std(rrilist),
        ", max, min:",
        np.max(rrilist),
        np.min(rrilist),
    )
    print(
        "   Risetime = ",
        np.mean(risetimelist),
        "+/-",
        np.std(risetimelist),
        ", max, min:",
        np.max(risetimelist),
        np.min(risetimelist),
    )
    print(
        "   Falltime = ",
        np.mean(falltimelist),
        "+/-",
        np.std(falltimelist),
        ", max, min:",
        np.max(falltimelist),
        np.min(falltimelist),
    )
    print(
        "   Height =   ",
        np.mean(heightlist),
        "+/-",
        np.std(heightlist),
        ", max, min:",
        np.max(heightlist),
        np.min(heightlist),
    )

    maxpeaklen = bisect.bisect_left(timeaxis, np.mean(rrilist) + 2.0 * np.std(rrilist))
    waveformsum = np.zeros((maxpeaklen), dtype="float")
    waveformwt = np.zeros((maxpeaklen), dtype="float")

    # now convert the peak list into waveforms
    indexstart = 0
    for index, thepeak in enumerate(filteredpeakdictlist):
        if usemin:
            peakloc = bisect.bisect_left(timeaxis, thepeak["starttime"])
            if peakloc < len(peakmarkers):
                peakmarkers[peakloc] = 1.0
        else:
            peakloc = bisect.bisect_left(
                timeaxis, thepeak["starttime"] + thepeak["risetime"]
            )
            if peakloc < len(peakmarkers):
                peakmarkers[peakloc] = 1.0

        rri = thepeak["rri"]
        if rri > 0:
            bpm = 60.0 / rri
        else:
            bpm = 0.0
        avglen = np.min([index, 20])
        timediff = (
            filteredpeakdictlist[index]["starttime"]
            - filteredpeakdictlist[index - avglen]["starttime"]
        )
        if timediff > 0.0:
            abpm = avglen * 60.0 / timediff
        else:
            abpm = 0.0
        filteredpeakdictlist[index]["avgbpm"] = abpm

        height = thepeak["height"]
        pkconfidence = thepeak["confidence"]
        indexend = bisect.bisect_left(timeaxis, thepeak["endtime"])
        for j in range(indexstart, indexend):
            rrinterval[j] = 1.0 * rri
            instbpm[j] = 1.0 * bpm
            avgbpm[j] = 1.0 * abpm
            peakheight[j] = 1.0 * height
            confidence[j] = 1.0 * pkconfidence

        # store the waveform data
        filteredpeakdictlist[index]["waveform"] = waveform[indexstart : indexend + 1]
        thislen = indexend - indexstart + 1
        copylen = np.min([thislen, maxpeaklen])
        if copylen > 0 and indexstart < len(timeaxis) - copylen - 1:
            waveformsum[0:copylen] += waveform[indexstart : indexstart + copylen]
            waveformwt[0:copylen] += (
                0.0 * waveform[indexstart : indexstart + copylen] + 1.0
            )

        # prepare for next peak
        indexstart = indexend

    # normalize the waveform
    waveformsum = np.nan_to_num(waveformsum / waveformwt)

    # finish out the waveforms
    if len(timeaxis) > indexend:
        rrinterval[indexend:] = rrinterval[indexend - 1]
        instbpm[indexend:] = instbpm[indexend - 1]
        avgbpm[indexend:] = avgbpm[indexend - 1]
        peakheight[indexend:] = peakheight[indexend - 1]
        confidence[indexend:] = confidence[indexend - 1]

    return (
        peakmarkers,
        rrinterval,
        instbpm,
        avgbpm,
        peakheight,
        confidence,
        waveformsum,
        filteredpeakdictlist,
    )


def locallpeaks(
    timeaxis, data, samplerate, winsizeinsecs=1.5, thresh=0.5, hysteresissecs=0.4
):
    peaklist = []
    rrinterval = np.zeros((len(data)), dtype="float")
    peaks = np.zeros((len(data)), dtype="float")
    instbpm = np.zeros((len(data)), dtype="float")
    lastpeakloc = timeaxis[0]
    lastrrinterval = 0.0
    indexstart = bisect.bisect_left(timeaxis, timeaxis[0] + winsizeinsecs / 2.0)
    firstpeak = -1
    for i in range(indexstart, len(data)):
        startpoint = max([0, bisect.bisect_left(timeaxis, timeaxis[i] - winsizeinsecs)])
        thepeakloc, thepeakht = rt_locpeak(
            timeaxis[startpoint:i],
            data[startpoint:i],
            samplerate,
            lastpeakloc,
            winsizeinsecs=winsizeinsecs,
            thresh=thresh,
        )
        peaklist.append(thepeakloc)
        if thepeakloc > 0.0:
            peakindex = min([bisect.bisect_right(timeaxis, thepeakloc), len(data) - 1])
            peaks[peakindex] = 1.0
            lastrrinterval = thepeakloc - lastpeakloc
            lastpeakloc = 1.0 * thepeakloc
            if firstpeak < 0:
                firstpeak = i
        rrinterval[i] = lastrrinterval
        if rrinterval[i] > 0.0:
            instbpm[i] = 60.0 / rrinterval[i]
    # fix the beginning
    rrinterval[:firstpeak] = rrinterval[firstpeak]
    instbpm[:firstpeak] = instbpm[firstpeak]

    return peaks, rrinterval, instbpm


def rt_locpeak(
    timeaxis,
    data,
    samplerate,
    lastpeaktime,
    starttime=0.0,
    winsizeinsecs=1.5,
    thresh=0.25,
    hysteresissecs=0.4,
    unipolar=True,
):
    # look at a limited time window
    currenttime = timeaxis[-1]
    startpoint = np.max([0, bisect.bisect_left(timeaxis, currenttime - winsizeinsecs)])
    lastpeakindex = np.max([0, bisect.bisect_left(timeaxis, lastpeaktime)])

    # find normative limits
    recentmax = np.max(data[startpoint:])
    recentmin = np.min(data[startpoint:])
    recentrange = recentmax - recentmin
    # print(currenttime-lastpeaktime,\
    #    recentmin,recentmax,recentrange,data[-1])

    # screen out obvious non-peaks
    if data[-1] < recentmin + recentrange * thresh:
        return -1.0, 0.0
    if currenttime - lastpeaktime < hysteresissecs:
        # print('too soon', currenttime, lastpeaktime)
        return -1.0, 0.0
    if np.min(data[lastpeakindex:]) > recentmin + recentrange * (1.0 - thresh):
        # print('non return to baseline')
        return -1.0, 0.0

    # now check to see if we have just passed a peak
    if data[-1] < data[-2]:
        if data[-2] <= data[-3]:
            fitstart = -5
            fitdata = data[fitstart:]
            X = (
                currenttime
                + (np.arange(0.0, len(fitdata)) - len(fitdata) + 1.0) / samplerate
            )
            maxtime = np.sum(X * fitdata) / np.sum(fitdata)
            maxsigma = np.sqrt(
                abs(np.sum((X - maxtime) ** 2 * fitdata) / np.sum(fitdata))
            )
            maxval = fitdata.max()
            peakheight, peakloc, peakwidth = gaussfit(
                maxval, maxtime, maxsigma, X, fitdata
            )
            if (abs(peakloc - maxtime) > 3.0 / samplerate) or peakloc > currenttime:
                # fit bombed
                print("fit failure - skipping peak")
                return -1.0, 0.0
            return peakloc, peakheight
        else:
            # print('not less enough', data[-3],data[-2])
            return -1.0, 0.0
    else:
        # print('not less', data[-2],data[-1]))
        return -1.0, 0.0


def parsetriggers(markerlist):
    mintr = 0.3
    maxtr = 10.0
    epochs = []

    previousmark = -100000.0
    for i in range(0, len(markerlist)):
        if maxtr > markerlist[i] - previousmark:
            # we are within a reasonable time of a previous pulse
            if markerlist[i] - previousmark > mintr:
                epochs[-1][1] = markerlist[i]
                epochs[-1][2] += 1
                epochs[-1][3] = (epochs[-1][1] - epochs[-1][0]) / (epochs[-1][2] - 1)
        else:
            # we've gone longer than a reasonable TR - this is a new series
            epochs.append([markerlist[i], markerlist[i], 1, 0.0])
        previousmark = markerlist[i]
    return epochs


def filterpeaklist(peaklist, debug=False):
    # Implemention of the algorithm from Logier, R, et al. "An efficient algorithm for R-R
    #     intervals series filtering", in v2 p3937-40, Proc. IEEE EMBS 2004.
    #
    # initialize the rri list with rational values
    # 99%	2.576
    # 98%	2.326
    # 95%	1.96
    # 90%	1.645
    numsds = 1.96  # 95% confidence interval
    m20 = 60.0 / 72.0
    s20 = m20 / 10.0
    rrilist = [np.random.normal(loc=m20, size=s20) for i in range(20)]
    rrilist = [elem["rri"] for elem in peaklist]
    m20 = np.median(rrilist)
    s20 = np.std(rrilist)
    rrilist = [elem["rri"] for elem in peaklist[:20]]

    # first go through and tag all the bad peaks
    isquestionable = False
    firstquestionable = 0
    print("peak filter detection pass")
    debuginfo = []
    for i in range(len(peaklist) - 1):
        # see if we are in the 95% range:
        c1 = False
        c2 = False
        c3 = False
        c4 = False
        c5 = False
        c6 = False
        c7 = False
        if (m20 - numsds * s20) <= peaklist[i]["rri"] <= (m20 + numsds * s20):
            # yes we are, so mark all peaks since last questionable one as good and update stats
            if debug:
                print(i, " is good")
            if isquestionable:
                for j in range(firstquestionable, i + 1):
                    peaklist[j][
                        "confidence"
                    ] = 0.5  # if we're here, the peak was questionable,
                    # but we now believe it's good
            else:
                peaklist[i]["confidence"] = 1.0  # if we're here, the peak is just good

            # update the stats to reflect the new good points
            for j in range(firstquestionable, i + 1):
                rrilist.append(peaklist[j]["rri"])
                rrilist.pop(0)
            m20 = np.mean(rrilist)
            s20 = np.std(rrilist)

            # reset the questionable peak detector
            firstquestionable = i + 1
            isquestionable = False
        else:
            # we are out of the 95% confidence range, so now we do the criterion checks
            # missed beat
            c1 = (peaklist[i]["rri"] < m20 - numsds * s20) and (
                peaklist[i + 1]["rri"] > m20 + numsds * s20
            )
            # spurious beat
            c2 = (peaklist[i]["rri"] < 0.75 * peaklist[i - 1]["rri"]) and (
                peaklist[i + 1]["rri"] < 0.75 * peaklist[i - 1]["rri"]
            )
            # ectopic beat
            c3 = peaklist[i]["rri"] > 1.75 * peaklist[i - 1]["rri"]

            # just insane numbers
            c4 = peaklist[i]["rri"] < 0.25
            c5 = peaklist[i]["rri"] > 1.6
            # c6 = peaklist[i]['risetime'] < 0.1
            # c7 = peaklist[i]['falltime'] < 0.3

            if c1 or c2 or c3 or c4 or c5 or c6 or c7:
                # this sample is clearly wrong - mark it and all questionable samples as wrong
                if debug:
                    print(i, " is bad")
                for j in range(firstquestionable, i + 1):
                    peaklist[j]["confidence"] = -1.0
                firstquestionable = i + 1
                isquestionable = False
            else:
                if not isquestionable:
                    isquestionable = True
                    firstquestionable = i + 0
        peaklist[i]["testresults"] = [
            c1 and True,
            c2 and True,
            c3 and True,
            c4 and True,
            c5 and True,
        ]
        peaklist[i]["m20"] = 1.0 * m20
        peaklist[i]["s20"] = 1.0 * s20
        peaklist[i]["firstquestionable"] = firstquestionable + 0
        peaklist[i]["isquestionable"] = isquestionable and True
    for i in range(len(peaklist) - 1):
        debuginfo.append(
            (
                i,
                peaklist[i]["starttime"],
                peaklist[i]["rri"],
                peaklist[i]["m20"],
                peaklist[i]["s20"],
                peaklist[i]["firstquestionable"],
                peaklist[i]["isquestionable"],
                peaklist[i]["testresults"][0],
                peaklist[i]["testresults"][1],
                peaklist[i]["testresults"][2],
                peaklist[i]["testresults"][3],
                peaklist[i]["testresults"][4],
                peaklist[i]["confidence"],
            )
        )

    if debug:
        for entry in debuginfo:
            print(entry)
    debuginfo_scanpass = pd.DataFrame.from_records(
        debuginfo,
        columns=[
            "peaknum",
            "starttime",
            "rri",
            "m20",
            "s20",
            "firstquestionable",
            "isquestionable",
            "c1",
            "c2",
            "c3",
            "c4",
            "c5",
            "confidence",
        ],
    )
    debuginfo_scanpass.to_csv("debuginfo_scanpass.csv")

    # now clean up the peak list
    debuginfo = []
    print("peak filter rebuilding pass")
    filteredpeaklist = []
    sourcept = 0
    while sourcept < len(peaklist):
        if peaklist[sourcept]["confidence"] < 0 and len(filteredpeaklist) > 0:
            t0 = filteredpeaklist[-1]["starttime"]  # t0 is time of last known good peak
            numbad = 0
            while (peaklist[sourcept + numbad]["confidence"] < 0) and (
                (sourcept + numbad) < len(peaklist) - 1
            ):
                numbad += 1
            tdiff = (
                peaklist[sourcept + numbad + 1]["starttime"]
                - filteredpeaklist[-1]["starttime"]
            )
            if debug:
                print("time gap is ", tdiff)
            avrri = (
                peaklist[sourcept + numbad + 1]["rri"] + filteredpeaklist[-1]["rri"]
            ) / 2.0
            numrebuilt = int(tdiff / avrri)
            if debug:
                print("average rri is ", avrri, ": ", numrebuilt, " points")
            rrislope = (
                peaklist[sourcept + numbad]["rri"] - filteredpeaklist[-1]["rri"]
            ) / tdiff
            risetimeslope = (
                peaklist[sourcept + numbad]["risetime"]
                - filteredpeaklist[-1]["risetime"]
            ) / tdiff
            falltimeslope = (
                peaklist[sourcept + numbad]["falltime"]
                - filteredpeaklist[-1]["falltime"]
            ) / tdiff
            heightslope = (
                peaklist[sourcept + numbad]["height"] - filteredpeaklist[-1]["height"]
            ) / tdiff
            bldiffslope = (
                peaklist[sourcept + numbad]["bldiff"] - filteredpeaklist[-1]["bldiff"]
            ) / tdiff
            if debug:
                print(
                    "patching",
                    numbad,
                    "points at",
                    sourcept,
                    "with",
                    numrebuilt,
                    "samples",
                )
                print(
                    "    start:",
                    filteredpeaklist[-1]["starttime"],
                    filteredpeaklist[-1]["rri"],
                )
            for i in range(0, numrebuilt):
                newentry = {}
                newentry["starttime"] = (
                    filteredpeaklist[-1]["starttime"] + filteredpeaklist[-1]["rri"]
                )
                newentry["rri"] = (
                    rrislope * (newentry["starttime"] - t0)
                    + filteredpeaklist[-1]["rri"]
                )
                newentry["endtime"] = newentry["starttime"] + newentry["rri"]
                newentry["risetime"] = (
                    risetimeslope * (newentry["starttime"] - t0)
                    + filteredpeaklist[-1]["risetime"]
                )
                newentry["falltime"] = (
                    falltimeslope * (newentry["endtime"] - t0)
                    - filteredpeaklist[-1]["risetime"]
                )
                newentry["height"] = (
                    heightslope * (newentry["starttime"] - t0)
                    + filteredpeaklist[-1]["height"]
                )
                newentry["bldiff"] = (
                    bldiffslope * (newentry["starttime"] - t0)
                    + filteredpeaklist[-1]["bldiff"]
                )
                newentry["confidence"] = 0.25
                filteredpeaklist.append(newentry)
                debuginfo.append(("RECON", newentry["starttime"], newentry["rri"]))
                if debug:
                    print("         :", newentry["starttime"], newentry["rri"])
            if debug:
                print(
                    "      end:",
                    peaklist[sourcept + numbad]["starttime"],
                    peaklist[sourcept + numbad]["rri"],
                )
            sourcept += numbad
        else:
            filteredpeaklist.append(peaklist[sourcept])
            if peaklist[sourcept]["confidence"] == 1.0:
                debuginfo.append(
                    (
                        "VALID",
                        peaklist[sourcept]["starttime"],
                        peaklist[sourcept]["rri"],
                    )
                )
            else:
                debuginfo.append(
                    (
                        "QUEST",
                        peaklist[sourcept]["starttime"],
                        peaklist[sourcept]["rri"],
                    )
                )
            sourcept += 1

    if debug:
        for entry in debuginfo:
            print(entry)
    debuginfo_reconpass = pd.DataFrame.from_records(
        debuginfo, columns=["peaknum", "starttime", "rri"]
    )
    debuginfo_reconpass.to_csv("debuginfo_reconpass.csv")

    # calculate and display summary statistics

    return filteredpeaklist


def peakdet(v, delta_y, delta_x_rise=0.0, delta_x_fall=0.0, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        print(len(v), "!=", len(x))
        sys.exit("Input vectors v and x must have same length")

    if not np.isscalar(delta_x_fall):
        sys.exit("Input argument delta_x_fall must be a scalar")

    if not np.isscalar(delta_x_rise):
        sys.exit("Input argument delta_x_rise must be a scalar")

    if delta_x_fall < 0:
        sys.exit("Input argument delta_x_fall must be non-negative")

    if delta_x_rise < 0:
        sys.exit("Input argument delta_x_rise must be non-negative")

    if not np.isscalar(delta_y):
        sys.exit("Input argument delta_y must be a scalar")

    if delta_y <= 0:
        sys.exit("Input argument delta_y must be positive")

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    lastmnpos, lastmxpos = 0, 0

    lookformax = False

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this <= mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if (this < mx - delta_y) and (x[i] - lastmnpos >= delta_x_rise):
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lastmxpos = x[i]
                lookformax = False
        else:
            if (this > mn + delta_y) and (x[i] - lastmxpos >= delta_x_fall):
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lastmnpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


# NP: Making a peak detector object so that each device can have a reference to its own peak detector,
# and thus multiple individuals' peaks are not all working off of same data.
class RTPeakDetector(object):
    def __init__(self):
        self.maxtab = []
        self.mintab = []
        self.mn, self.mx = np.Inf, -np.Inf
        self.mnpos, self.mxpos = np.NaN, np.NaN
        self.lastmnpos, self.lastmxpos = 0, 0
        self.indexval = 0

        self.lookformax = False

    def rt_peakdet(self, v, delta_y, delta_x_rise=0.0, delta_x_fall=0.0, x=None):
        """
        Converted from MATLAB script at http://billauer.co.il/peakdet.html

        Returns two arrays

        function [maxtab, mintab]=peakdet(v, delta_y, x)
        %PEAKDET Detect peaks in a vector
        %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
        %        maxima and minima ("peaks") in the vector V.
        %        MAXTAB and MINTAB consists of two columns. Column 1
        %        contains indices in V, and column 2 the found values.
        %
        %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
        %        in MAXTAB and MINTAB are replaced with the corresponding
        %        X-values.
        %
        %        A point is considered a maximum peak if it has the maximal
        %        value, and was preceded (to the left) by a value lower by
        %        DELTA.

        % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
        % This function is released to the public domain; Any use is allowed.

        """

        if x is None:
            x = self.indexval

        if not np.isscalar(delta_y):
            sys.exit("Input argument delta_y must be a scalar")

        if not np.isscalar(delta_x_fall):
            sys.exit("Input argument delta_x_fall must be a scalar")

        if not np.isscalar(delta_x_rise):
            sys.exit("Input argument delta_x_rise must be a scalar")

        if delta_x_fall < 0:
            sys.exit("Input argument delta_x_fall must be non-negative")

        if delta_x_rise < 0:
            sys.exit("Input argument delta_x_rise must be non-negative")

        if delta_y <= 0:
            sys.exit("Input argument delta_y must be positive")

        this = v
        if this > self.mx:
            self.mx = this
            self.mxpos = x
        if this <= self.mn:
            self.mn = this
            self.mnpos = x

        isamax = False
        isamin = False
        if self.lookformax:
            if (this < self.mx - delta_y) and (x - self.lastmnpos >= delta_x_rise):
                self.maxtab.append((self.mxpos, self.mx))
                isamax = True
                self.mn = this
                self.mnpos = x
                self.lookformax = False
        else:
            if (this > self.mn + delta_y) and (x - self.lastmxpos >= delta_x_fall):
                self.mintab.append((self.mnpos, self.mn))
                isamin = True
                self.mx = this
                self.mxpos = x
                self.lookformax = True

        self.indexval += 1

        return isamax, isamin


# to call:
def init_rt_peakdet():
    global maxtab, mintab
    global mn, mx, mnpos, mxpos
    global lastmnpos, lastmxpos
    global lookformax
    global indexval

    maxtab = []
    mintab = []
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN
    lastmnpos, lastmxpos = 0, 0
    indexval = 0

    lookformax = False


# def rt_peakdet(v, delta_y, delta_x_rise=0.0, delta_x_fall=0.0, x=None):
#     """
#     Converted from MATLAB script at http://billauer.co.il/peakdet.html
#
#     Returns two arrays
#
#     function [maxtab, mintab]=peakdet(v, delta_y, x)
#     %PEAKDET Detect peaks in a vector
#     %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
#     %        maxima and minima ("peaks") in the vector V.
#     %        MAXTAB and MINTAB consists of two columns. Column 1
#     %        contains indices in V, and column 2 the found values.
#     %
#     %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
#     %        in MAXTAB and MINTAB are replaced with the corresponding
#     %        X-values.
#     %
#     %        A point is considered a maximum peak if it has the maximal
#     %        value, and was preceded (to the left) by a value lower by
#     %        DELTA.
#
#     % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
#     % This function is released to the public domain; Any use is allowed.
#
#     """
#     global maxtab, mintab
#     global mn, mx, mnpos, mxpos
#     global lastmnpos, lastmxpos
#     global lookformax
#     global indexval
#
#     if x is None:
#         x = indexval
#
#     if not np.isscalar(delta_y):
#         sys.exit('Input argument delta_y must be a scalar')
#
#     if not np.isscalar(delta_x_fall):
#         sys.exit('Input argument delta_x_fall must be a scalar')
#
#     if not np.isscalar(delta_x_rise):
#         sys.exit('Input argument delta_x_rise must be a scalar')
#
#     if delta_x_fall < 0:
#         sys.exit('Input argument delta_x_fall must be non-negative')
#
#     if delta_x_rise < 0:
#         sys.exit('Input argument delta_x_rise must be non-negative')
#
#     if delta_y <= 0:
#         sys.exit('Input argument delta_y must be positive')
#
#     this = v
#     if this > mx:
#         mx = this
#         mxpos = x
#     if this <= mn:
#         mn = this
#         mnpos = x
#
#     isamax = False
#     isamin = False
#     if lookformax:
#         if (this < mx - delta_y) and (x - lastmnpos >= delta_x_rise):
#             maxtab.append((mxpos, mx))
#             isamax = True
#             mn = this
#             mnpos = x
#             lookformax = False
#     else:
#         if (this > mn + delta_y) and (x - lastmxpos >= delta_x_fall):
#             mintab.append((mnpos, mn))
#             isamin = True
#             mx = this
#             mxpos = x
#             lookformax = True
#
#     indexval += 1
#
#     return isamax, isamin


"""
if __name__=="__main__":
    from matplotlib.pyplot import plot, scatter, show
    series = [0,0,0,2,0,0,0,-2,0,0,0,2,0,0,0,-2,0]
    maxtab, mintab = peakdet(series,.3)
    plot(series)
    scatter(np.array(maxtab)[:,0], np.array(maxtab)[:,1], color='blue')
    scatter(np.array(mintab)[:,0], np.array(mintab)[:,1], color='red')
    init_rt_peakdet()
    rt_maxtab = []
    rt_mintab = []
    for i in range(0, len(series)):
        isamax, isamin = rt_peakdet(series[i], 0.3)
        if isamax:
            rt_maxtab.append((i, series[i]))
        if isamin:
            rt_mintab.append((i, series[i]))
    offset = 1.5
    plot(np.array(series) + offset)
    scatter(np.array(maxtab)[:,0], np.array(maxtab)[:,1] + offset, color='blue')
    scatter(np.array(mintab)[:,0], np.array(mintab)[:,1] + offset, color='red')
    show()
"""
