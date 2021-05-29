# ----------------------------------------------------------------------------------------------------
# ProcessData.py
# Authors: Fernando Dias <fernandodias@gta.ufrj.br>, Matheus Felinto <felinto@gta.ufrj.br>
# 
# Description:
#
# Usage:
#
# ----------------------------------------------------------------------------------------------------


import os
import io
import sys
import json
import time
import pickle
import signal
import datetime
import argparse
import pathlib
from getopt import getopt
from configparser import ConfigParser

import numpy as np
import pandas as pd
import psycopg2 as dblib

from data_processing import LineCorrection, LineDetection
from TimeMeasure import Measure
from logger import logger


def main():
    # --------------------------------
    # Setup phase
    # --------------------------------
    performanceData = dict()
    mes = Measure()

    # Parse configurations and parameters, sets logger and global variables
    CONFIGS = ConfigParser()
    
    # TODO: Remove this comment when assured it's not needed anymore
    # CONFIGS['lineDetection']['busStepSize'] = '5'
    # CONFIGS['lineDetection']['lineStepSize'] = '5'
    # CONFIGS['lineDetection']['distanceTolerance'] = '300'
    # CONFIGS['lineDetection']['detectionPercentage'] = '0.9'
    # CONFIGS.add_section('lineCorrection')
    # CONFIGS['lineCorrection']['limit'] = '3'

    busFilter = None
    lineFilter = None
    desiredDate = None

    parser = argparse.ArgumentParser(description="Trajectory classifier with GPU acceleration. Classifies long bus paths by line.")

    parser.add_argument('-c','--config',default=None,type=pathlib.Path,help="New configuration path. Overrides default path.")
    parser.add_argument("-d","-date",type=datetime.date.fromisoformat,default=None,help="(YYYY-MM-DD) Date to query bus paths")
    parser.add_argument("-b","--bus-whitelist",type=pathlib.Path,help="Path to bus identifier's whitelist. Either comma or line separated. Only those IDs will be parsed.")
    parser.add_argument("-l",'--line-whitelist',type=pathlib.Path,help="Path to line identifier's whitelist. Either comma or line separated. Only those IDs will be parsed.")
    parser.add_argument("--bus-blacklist",type=pathlib.Path,help="Path to bus identifier's whitelist. Either comma or line separated. Those IDs will be excluded from list.")
    parser.add_argument("--line-blacklist",type=pathlib.Path,help="Path to line identifier's whitelist. Either comma or line separated. Those IDs will be excluded from list.")
    parser.add_argument("-e","--everything",action="store_true",help="Flag to allow program to process all data without any filter on either bus or lines. Required when no black- or whitelist is given.")
    parser.add_argument("-v",'--verbose',action="count",default=0,help="Increase output verbosity")

    args = parser.parse_args()

    # Sets logging details
    log = logger(args.verbose)

    # sets configuration options
    configPath = pathlib.Path("/var/secrets")
    if (configPath / 'main.conf').exists():
            CONFIGS.read(configPath / 'main.conf')

    if args.config:
        if not args.config.exists():
            log.critical(f"Configuration file at {args.config.absolute()} does not exits")
            exit(1)
        CONFIGS.read(args.config)

    # Get execution parameters
    desiredDate = args.date
    lineFilter = args.line_whitelist
    busFilter = args.bus_whitelist

    if not desiredDate:
        log.critical("No filters or desired date")
        exit(1)

    if not args.everything and (lineFilter is None and busFilter is None):
        log.critical("No filters where set and -e option was not set. Program will not operate on all the data unless explicitly said so.")
        exit(1)
    
    # ------------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------------

    mes.start("data-acquisition")
    nextDate = desiredDate + datetime.timedelta(days=1)
    
    # Starting database connection
    database = dblib.connect(**CONFIGS['database'])

    # Getting both bus and lines Ids as lists, ordered by size
    # Bus Ids
    with database.cursor() as cursor:
            cursor.execute(f"SELECT bus_id,COUNT(time_detection) AS points FROM bus_data WHERE time_detection BETWEEN %s::timestamp AND %s::timestamp GROUP BY bus_id ORDER BY points DESC",(desiredDate,nextDate))
            busSizeListComplete = cursor.fetchall()
    busMaxSize = busSizeListComplete[0][1]
    
    # line Ids
    with database.cursor() as cursor:
            cursor.execute("SELECT line_id,direction,COUNT(position) as dist FROM line_data_simple GROUP BY line_id,direction ORDER BY dist DESC")
            lineSizeListComplete = cursor.fetchall()
    lineMaxSize = lineSizeListComplete[0][2]

    # Buses and lines Tables creation and filter processing
    busMatrix = None # R3 matrix of bus x lat x lon
    busTimestamps = dict()
    busSizeList = list()

    # TODO: Poorly implemented, should be 1 request with filters sent
    for busId,size in busSizeListComplete:
        if busFilter:
            if not busId in busFilter:
                continue
        busSizeList += [(busId,size)]
        with database.cursor() as cursor:
            cursor.execute("""SELECT time_detection,latitude,longitude FROM bus_data 
                WHERE bus_id=%s AND time_detection BETWEEN %s AND %s  ORDER BY time_detection""",(busId,desiredDate,nextDate))
            queryResults = cursor.fetchall()
        curTime = [ (i[0]) for i in queryResults ]
        busTimestamps[busId] = curTime
        currentBusPath = np.array([ (i[1],i[2]) for i in queryResults ])
        currentBusPath = np.pad(currentBusPath,((0,busMaxSize - len(currentBusPath)),(0,0)),constant_values=np.nan)
        currentBusPath = np.expand_dims(currentBusPath,axis=0)
            
        if busMatrix is None:
            busMatrix = np.copy(currentBusPath)
        else:
            busMatrix = np.concatenate([busMatrix,currentBusPath],axis=0)


    lineMatrix = None 
    lineSizeList = list()
    for lineId,direction,size in lineSizeListComplete:
        if lineFilter:
            if not lineId in lineFilter:
                continue
        lineSizeList += [(lineId,direction,size)]
        with database.cursor() as cursor:
            cursor.execute("""SELECT latitude,longitude FROM line_data_simple 
                WHERE line_id=%s AND direction=%s""",(lineId,direction))
            currentLinePath = np.array(cursor.fetchall())
            currentLinePath = np.pad(currentLinePath,((0,lineMaxSize - len(currentLinePath)),(0,0)),constant_values=np.nan)
            currentLinePath = np.expand_dims(currentLinePath,axis=0)

        if lineMatrix is None:
            lineMatrix = np.copy(currentLinePath)
        else:
            lineMatrix = np.concatenate([lineMatrix,currentLinePath],axis=0)


    if lineMatrix is None or busMatrix is None:
        raise Exception("Line or Bus match no bus")
    
    mes.end("data-acquisition")

    # -------------------
    # Statistical data and temporary save of files
    # -------------------

    # Good statistical data for measure later on
    # Could be part of TimeMeasure:Measure class, so it can save itself without the use of signals
    performanceData = dict()
    performanceData['bus-amount'] = len(busSizeList)
    performanceData['line-amount'] = len(lineSizeList)
    performanceData['total-line-coordinates'] = sum([int(i[2]) for i in lineSizeList])
    performanceData['total-bus-coordinates'] = sum([int(i[1]) for i in busSizeList])
    performanceData['bus-matrix-max-points'] = busMatrix.shape[1]
    performanceData['line-matrix-max-points'] = busMatrix.shape[1]
    performanceData['bus-wasted-points'] = ((busMatrix.shape[0]*busMatrix.shape[1])-performanceData['total-bus-coordinates'])
    performanceData['line-wasted-points'] = ((lineMatrix.shape[0]*lineMatrix.shape[1])-performanceData['total-line-coordinates'])
    performanceData['total-size-bus-matrix-mbytes'] = (32*2*busMatrix.shape[0]*busMatrix.shape[1])/(8*10**6)
    performanceData['total-size-line-matrix-mbytes'] = (32*2*lineMatrix.shape[0]*lineMatrix.shape[1])/(8*10**6)
    performanceData['extrapolated-maximum-d-size-mbytes'] = (32*busMatrix.shape[0]*busMatrix.shape[1]*lineMatrix.shape[0]*lineMatrix.shape[1])/(8*10**6)
    performanceData['maximum-d-size-batched-mbytes'] = (32*int(CONFIGS['lineDetection']['busStepSize'])*busMatrix.shape[1]*int(CONFIGS['lineDetection']['lineStepSize'])*lineMatrix.shape[1])/(8*10**6)
    performanceData['iterations-amount'] = (performanceData['bus-amount']/int(CONFIGS['lineDetection']['busStepSize']))*(performanceData['line-amount']/int(CONFIGS['lineDetection']['lineStepSize']))
    
    # Saving temporary data in case of crash for fast recovery (not implemented yet TODO)
    with open("tmp/busSizeList.pickle",'wb') as fil:
        pickle.dump(busSizeList,fil)

    with open("tmp/lineSizeList.pickle",'wb') as fil:
        pickle.dump(lineSizeList,fil)

    with open("tmp/busMatrix.pickle",'wb') as fil:
        pickle.dump(busMatrix,fil)

    with open("tmp/lineMatrix.pickle",'wb') as fil:
        pickle.dump(lineMatrix,fil)

    with open("tmp/busTimestamps.pickle",'wb') as fil:
        pickle.dump(busTimestamps,fil)


    # -------------------
    # Line detection and correction phase
    # -------------------

    # Line detection phase
    mes.start("line-detection")

    mes.start("line-detection-function")
    detectionMatrix = LineDetection.FilterData(busMatrix,lineMatrix,busSizeList,lineSizeList,CONFIGS,log)
    mes.end("line-detection-function")
    
    mes.start('line-detection-saving')
    with open("tmp/detectMatrix.pickle.tmp",'wb') as fil:
        pickle.dump(detectionMatrix,fil)
    mes.end("line-detection-saving")
    
    mes.end("line-detection")


    # Line correction phase
    mes.start("line-correction")

    mes.start('line-correction-function')
    busResultTable = LineCorrection.CorrectData((detectionMatrix > float(CONFIGS['lineDetection']['detectionPercentage'])),busMatrix,lineMatrix,busSizeList,lineSizeList,CONFIGS) # Matriz R^3 de (entidade x indice)
    mes.end("line-correction-function")

    mes.start('line-correction-saving')
    with open("tmp/busResultTable.pickle.tmp",'wb') as fil:
        pickle.dump(busResultTable,fil)
    mes.end("line-correction-saving")
    
    mes.end("line-correction")
    
    # ----------------------------------------------------------------------------
    # Database saving
    # -----------------------------------------------------------------------------

    # TODO: Change write to DB method, too slow
    mes.start('database-insertion')
    # Update bus table
    databaseInsert = list()
    for _, currentBusIndex in enumerate(busResultTable.columns):
        currentBusTimeDetection = busTimestamps[currentBusIndex]
        currentBusCorrectionResult = busResultTable[currentBusIndex]
        databaseInsert += [(currentBusTimeDetection[i],currentBusIndex,currentBusCorrectionResult[i],currentBusIndex) for i in range(len(currentBusTimeDetection))]
    with database.cursor() as cursor:
        cursor.executemany("""INSERT INTO 
            bus_data(time_detection,bus_id,line_key_detected) 
                VALUES (%s,%s,%s)
        ON CONFLICT (time_detection,bus_id) DO UPDATE SET line_key_detected = EXCLUDED.line_key_detected
        WHERE 
            bus_data.time_detection BETWEEN '2019-05-06' AND '2019-05-07' AND
            bus_data.bus_id=%s""",databaseInsert)
        database.commit()
    mes.end('database-insertion')
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Bus trajectory classifier with GPU acceleration")

    parser.add_argument('-c','--config',type=pathlib.Path,help="New configuration path. Overrides default path.")
    parser.add_argument("-d","-date",type=datetime.date.fromisoformat,help="(YYYY-MM-DD) Date to query bus paths")
    parser.add_argument("-b","--bus-whitelist",type=pathlib.Path,help="Path to bus identifier's whitelist. Either comma or line separated. Only those IDs will be parsed.")
    parser.add_argument("-l",'--line-whitelist',type=pathlib.Path,help="Path to line identifier's whitelist. Either comma or line separated. Only those IDs will be parsed.")
    parser.add_argument("--bus-blacklist",type=pathlib.Path,help="Path to bus identifier's whitelist. Either comma or line separated. Those IDs will be excluded from list.")
    parser.add_argument("--line-blacklist",type=pathlib.Path,help="Path to line identifier's whitelist. Either comma or line separated. Those IDs will be excluded from list.")
    parser.add_argument("-e","--everything",action="store_true",help="Flag to allow program to process all data without any filter on either bus or lines. Required when no black- or whitelist is given.")
    parser.add_argument("")
    main()
    