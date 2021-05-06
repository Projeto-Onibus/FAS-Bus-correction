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
import logging
import datetime
from getopt import getopt
from configparser import ConfigParser

import numpy as np
import pandas as pd
import psycopg2 as dblib

from data_processing import LineCorrection, LineDetection
from TimeMeasure import Measure

# TODO: Remove global variables implementation, use logging
performanceData = dict()
mes = Measure()

# def print_to_string(*args, **kwargs):
#     output = io.StringIO()
#     print(*args, file=output, **kwargs)
#     contents = output.getvalue()
#     output.close()
#     return contents

# def GracefulExit(signal,term):
#     global performanceData,mes
#     performanceData['time'] = print_to_string(mes)
#     with open("tmp/perfdata.json",'w') as fil:
#         json.dump(performanceData,fil)
#     exit(0)

#
def main():
    # --------------------------------
    # Setup phase
    # --------------------------------

    # Parse configurations and parameters, sets logger and global variables
    CONFIGS = ConfigParser()
    log = logging
    global performanceData,mes
    
    # TODO: use argparse to beter handle cli interaction
    for filename in [i for i in os.listdir() if i[-4:] == ".ini"]:
        CONFIGS.read(filename)
    smallOpts = ['o:']
    bigopts = ["bus-filter=","line-filter=","desired-date=","options-file=","stop-when="]
    options, _ = getopt(sys.argv[1:],smallOpts,bigopts)

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

    for option in options:
        if option[0] == "-o" or option[0] == "--options-file":
            CONFIGS.read(option[1])
        elif option[0] == '--bus-filter':
            with open(option[1],'r') as fi:
                busFilter = [i for i in fi.read().split(",") if len(i)>0]
        elif option[0] == '--line-filter':
            with open(option[1],'r') as fi:
                lineFilter = [i for i in fi.read().split(",") if len(i)>0]
        elif option[0] == '--desired-date':
                desiredDate = datetime.datetime.strptime(option[1],"%Y-%m-%d")

    if not desiredDate:
        logging.critical("No filters or desired date")
        exit(1)

    if not busFilter and not lineFilter:
        raise Exception("Too much to handle")
    
    # ------------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------------

    mes.start("data-acquisition")
    nextDate = desiredDate + datetime.timedelta(days=1)
    
    # Starting database connection
    database = dblib.connect(**CONFIGS['database'])

    # Getting both bus and lines Ids as lists, ordered by size
    mes.start("get-lists")
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
    mes.end("get-lists")

    mes.start("get-bus-data")
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

    mes.end("get-bus-data")
    mes.start("get-line-data")

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
    
    mes.end("get-line-data")
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
    detectionMatrix = LineDetection.FilterData(busMatrix,lineMatrix,busSizeList,lineSizeList,CONFIGS,logging)
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
    for position, currentBusIndex in enumerate(busResultTable.columns):
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
    main()