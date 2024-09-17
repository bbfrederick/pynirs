#!/usr/bin/env python
#
# $Author: frederic $
# $Date: 2016/03/18 18:54:44 $
# $Id: cbv_funcs.py,v 1.1.1.1 2016/03/18 18:54:44 frederic Exp $
#
import sys
import os

import numpy as np
import pandas as pd
import json


def makecolname(colnum, startcol):
    return f"col_{str(colnum + startcol).zfill(2)}"


def readbidstsv(inputfilename, colspec=None, warn=True, debug=False):
    r"""Read time series out of a BIDS tsv file

    Parameters
    ----------
    inputfilename : str
        The root name of the tsv and accompanying json file (no extension)
    colspec: list
        A comma separated list of column names to return
    debug : bool
        Output additional debugging information

    Returns
    -------
        samplerate : float
            Sample rate in Hz
        starttime : float
            Time of first point, in seconds
        columns : str array
            Names of the timecourses contained in the file
        data : 2D numpy array
            Timecourses from the file

    NOTE:  If file does not exist or is not valid, all return values are None

    """
    thefileroot, theext = os.path.splitext(inputfilename)
    if theext == ".gz":
        thefileroot, thenextext = os.path.splitext(thefileroot)
        if thenextext is not None:
            theext = thenextext + theext

    if debug:
        print("thefileroot:", thefileroot)
        print("theext:", theext)
    if os.path.exists(thefileroot + ".json") and (
        os.path.exists(thefileroot + ".tsv.gz") or os.path.exists(thefileroot + ".tsv")
    ):
        with open(thefileroot + ".json", "r") as json_data:
            d = json.load(json_data)
            try:
                samplerate = float(d["SamplingFrequency"])
            except:
                print("no samplerate found in json, setting to 1.0")
                samplerate = 1.0
                if warn:
                    print(
                        "Warning - SamplingFrequency not found in "
                        + thefileroot
                        + ".json.  This is not BIDS compliant."
                    )
            try:
                starttime = float(d["StartTime"])
            except:
                print("no starttime found in json, setting to 0.0")
                starttime = 0.0
                if warn:
                    print(
                        "Warning - StartTime not found in "
                        + thefileroot
                        + ".json.  This is not BIDS compliant."
                    )
            try:
                columns = d["Columns"]
            except:
                if debug:
                    print(
                        "no columns found in json, will take labels from the tsv file"
                    )
                columns = None
                if warn:
                    print(
                        "Warning - Columns not found in "
                        + thefileroot
                        + ".json.  This is not BIDS compliant."
                    )
            else:
                columnsource = "json"
        if os.path.exists(thefileroot + ".tsv.gz"):
            compression = "gzip"
            theextension = ".tsv.gz"
        else:
            compression = None
            theextension = ".tsv"
            if warn:
                print(
                    "Warning - "
                    + thefileroot
                    + ".tsv is uncompressed.  This is not BIDS compliant."
                )

        df = pd.read_csv(
            thefileroot + theextension,
            compression=compression,
            names=columns,
            header=None,
            sep="\t",
            quotechar='"',
        )

        # replace nans with 0
        df = df.fillna(0.0)

        # check for header line
        if any(df.iloc[0].apply(lambda x: isinstance(x, str))):
            headerlinefound = True
            # reread the data, skipping the first row
            df = pd.read_csv(
                thefileroot + theextension,
                compression=compression,
                names=columns,
                header=0,
                sep="\t",
                quotechar='"',
            )

            # replace nans with 0
            df = df.fillna(0.0)

            if warn:
                print(
                    "Warning - Column header line found in "
                    + thefileroot
                    + ".tsv.  This is not BIDS compliant."
                )
        else:
            headerlinefound = False

        if columns is None:
            columns = list(df.columns.values)
            columnsource = "tsv"
        if debug:
            print(
                samplerate,
                starttime,
                columns,
                np.transpose(df.to_numpy()).shape,
                (compression == "gzip"),
                warn,
                headerlinefound,
            )

        # select a subset of columns if they were specified
        if colspec is None:
            return (
                samplerate,
                starttime,
                columns,
                np.transpose(df.to_numpy()),
                (compression == "gzip"),
                columnsource,
            )
        else:
            collist = colspec.split(",")
            try:
                selectedcols = df[collist]
            except KeyError:
                print("specified column list cannot be found in", inputfilename)
                return [None, None, None, None, None, None]
            columns = list(selectedcols.columns.values)
            return (
                samplerate,
                starttime,
                columns,
                np.transpose(selectedcols.to_numpy()),
                (compression == "gzip"),
                columnsource,
            )
    else:
        print("file pair does not exist")
        return [None, None, None, None, None, None]


def writebidstsv(
    outputfileroot,
    data,
    samplerate,
    extraheaderinfo=None,
    compressed=True,
    columns=None,
    starttime=0.0,
    append=False,
    colsinjson=True,
    colsintsv=False,
    omitjson=False,
    debug=False,
):
    """
    NB: to be strictly valid, a continuous BIDS tsv file (i.e. a "_physio" or "_stim" file) requires:
    1) The .tsv is compressed (.tsv.gz)
    2) "SamplingFrequency", "StartTime", "Columns" must exist and be in the .json file
    3) The tsv file does NOT have column headers.
    4) "_physio" or "_stim" has to be at the end of the name, although this seems a little flexible

    The first 3 are the defaults, but if you really want to override them, you can.

    :param outputfileroot:
    :param data:
    :param samplerate:
    :param compressed:
    :param columns:
    :param starttime:
    :param append:
    :param colsinjson:
    :param colsintsv:
    :param omitjson:
    :param debug:
    :return:
    """
    if debug:
        print("entering writebidstsv:")
        print("\toutputfileroot:", outputfileroot)
        print("\tdata.shape:", data.shape)
        print("\tsamplerate:", samplerate)
        print("\tstarttime:", starttime)
        print("\tcompressed:", compressed)
        print("\tcolumns:", columns)
        print("\tstarttime:", starttime)
        print("\tappend:", append)
    if len(data.shape) == 1:
        reshapeddata = data.reshape((1, -1))
        if debug:
            print("input data reshaped from", data.shape, "to", reshapeddata.shape)
    else:
        reshapeddata = data
    if append:
        insamplerate, instarttime, incolumns, indata, incompressed, incolsource = (
            readbidstsv(outputfileroot + ".json", debug=debug)
        )
        if debug:
            print("appending")
            print(
                insamplerate, instarttime, incolumns, indata, incompressed, incolsource
            )
        if insamplerate is None:
            # file does not already exist
            if debug:
                print("creating file:", data.shape, columns, samplerate)
            startcol = 0
        else:
            # file does already exist
            if debug:
                print(
                    "appending:",
                    insamplerate,
                    instarttime,
                    incolumns,
                    incolsource,
                    indata.shape,
                    reshapeddata.shape,
                )
            compressed = incompressed
            if (
                (insamplerate == samplerate)
                and (instarttime == starttime)
                and reshapeddata.shape[1] == indata.shape[1]
            ):
                startcol = len(incolumns)
            else:
                print("data dimensions not compatible with existing dimensions")
                print(samplerate, insamplerate)
                print(starttime, instarttime)
                print(columns, incolumns, incolsource)
                print(indata.shape, reshapeddata.shape)
                sys.exit()
    else:
        startcol = 0

    if columns is None:
        columns = []
        for i in range(reshapeddata.shape[0]):
            columns.append(makecolname(i, startcol))
    else:
        if len(columns) != reshapeddata.shape[0]:
            raise ValueError(
                f"number of column names ({len(columns)}) ",
                f"does not match number of columns ({reshapeddata.shape[0]}) in data",
            )
    if startcol > 0:
        df = pd.DataFrame(data=np.transpose(indata), columns=incolumns)
        for i in range(len(columns)):
            df[columns[i]] = reshapeddata[i, :]
    else:
        df = pd.DataFrame(data=np.transpose(reshapeddata), columns=columns)
    if compressed:
        df.to_csv(
            outputfileroot + ".tsv.gz",
            sep="\t",
            compression="gzip",
            header=colsintsv,
            index=False,
        )
    else:
        df.to_csv(
            outputfileroot + ".tsv",
            sep="\t",
            compression=None,
            header=colsintsv,
            index=False,
        )
    headerdict = {}
    headerdict["SamplingFrequency"] = float(samplerate)
    headerdict["StartTime"] = float(starttime)
    if colsinjson:
        if startcol == 0:
            headerdict["Columns"] = columns
        else:
            headerdict["Columns"] = incolumns + columns
    if extraheaderinfo is not None:
        for key in extraheaderinfo:
            headerdict[key] = extraheaderinfo[key]

    if not omitjson:
        if debug:
            print(f"headerdict: {headerdict}")
        with open(outputfileroot + ".json", "wb") as fp:
            fp.write(
                json.dumps(
                    headerdict, sort_keys=True, indent=4, separators=(",", ":")
                ).encode("utf-8")
            )
