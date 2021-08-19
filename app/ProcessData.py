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
import pickle
import datetime
import pathlib
from getopt import getopt
from configparser import ConfigParser

import numpy as np
import pandas as pd
import psycopg2 as dblib

from data_processing import LineCorrection, LineDetection
from utils.TimeMeasure import Measure
from utils.logger import logger
from utils.Parser import parse_args

def main():
	# --------------------------------
	# Setup phase
	# --------------------------------
	performanceData = dict()
	mes = Measure("performance.txt")

	# Parse configurations and parameters, sets logger and global variables
	CONFIGS = ConfigParser()

	busFilter = None
	lineFilter = None
	desiredDate = None

	args = parse_args()

	# Sets logging details
	log = logger(args.verbose)

	argsReturned = vars(args).keys()

	# sets configuration options
	configPath = pathlib.Path("/run/secrets")
	if (configPath / 'main_configurations').exists():
			CONFIGS.read(configPath / 'main_configurations')

	if args.config:
		if not args.config.exists():
			log.critical(f"Configuration file at {args.config.absolute()} does not exits")
			exit(1)
		CONFIGS.read(args.config)

	desiredDate = args.date    

	# Get execution parameters
	if not desiredDate:
		log.critical("No desired date")
		exit(1)

	# lineFilter = args.line_whitelist if "line_whitelist" in argsReturned else None
	# busFilter = args.bus_whitelist if "bus_whitelist" in argsReturned else None


	# ------------------------------------------------------------------------
	# Data acquisition
	# ------------------------------------------------------------------------

	mes.start("data-acquisition")
	nextDate = desiredDate + datetime.timedelta(days=1)
	
	# Starting database connection
	database = dblib.connect(**CONFIGS['database'])

	if args.status:
		print("Status: OK")
		exit(0)
	log.debug("Bus ids from database")

	# Getting both bus and lines Ids as lists, ordered by size
	# Bus Ids
	with database.cursor() as cursor:
			cursor.execute(f"SELECT bus_id,COUNT(time_detection) AS points FROM bus_data WHERE time_detection BETWEEN %s::timestamp AND %s::timestamp GROUP BY bus_id ORDER BY points DESC",(desiredDate,nextDate))
			busSizeListComplete = cursor.fetchall()

	if len(busSizeListComplete) == 0:
		raise Exception(f"No buses in database at desired date {desiredDate}")

	busMaxSize = busSizeListComplete[0][1]
	log.debug("line ids from database")
	
	# line Ids
	with database.cursor() as cursor:
			cursor.execute("SELECT line_id,direction,COUNT(position) as dist FROM line_data GROUP BY line_id,direction ORDER BY dist DESC")
			lineSizeListComplete = cursor.fetchall()

	if (lineSizeListComplete) == 0:
		raise Exception("No lines in database")
	lineMaxSize = lineSizeListComplete[0][2]

	log.debug("filtering by whitelist/blacklist")
	# Filtering based on whitelists and blacklists
	busCompleteSet =set([i[0] for i in busSizeListComplete])
	lineCompleteSet =set([i[0] for i in lineSizeListComplete])

	if args.whitelist_lines:
		with open(args.line_whitelist_path,"r") as lineWhitelistFile:
			lineWhitelist = set([i for i in lineWhitelistFile.read().split("\n") if len(i) > 0 and i[0] != '#'])
	else:
		lineWhitelist = lineCompleteSet

	if args.blacklist_lines:
		with open(args.line_blacklist_path,"r") as lineBlacklistFile:
			lineBlacklist = set([i for i in lineBlacklistFile.read().split("\n") if len(i) > 0 and i[0] != '#'])
	else:
		lineBlacklist = set()

	if args.whitelist_buses:
		with open(args.bus_whitelist_path,"r") as busWhitelistFile:
			busWhitelist = set([i for i in busWhitelistFile.read().split("\n") if len(i) > 0 and i[0] != '#'])
	else:
		busWhitelist = busCompleteSet

	if args.blacklist_buses:
		with open(args.bus_blacklist_path,"r") as busBlacklistFile:
			busBlacklist = set([i for i in busBlacklistFile.read().split("\n") if len(i) > 0 and i[0] != '#'])
	else:
		busBlacklist = set()

	lineSet = lineCompleteSet.intersection(lineWhitelist).difference(lineBlacklist)
	busSet = busCompleteSet.intersection(busWhitelist).difference(busBlacklist)

	busSizeList = [i for i in busSizeListComplete if i[0] in busSet]
	lineSizeList = [i for i in lineSizeListComplete if i[0] in lineSet]

	if not args.everything and not (args.whitelist_buses or args.whitelist_lines or args.blacklist_buses or args.blacklist_lines):
		log.critical("No filters where set and -e option was not set. Program will not operate on all the data unless explicitly said so.")
		exit(1)
	
	log.info(f"Starting comparison between {len(busSizeList)} buses and {len(lineSizeList)} lines")
	log.debug("Requesting bus data from database")
	# Buses and lines Tables creation
	busMatrix = np.full((len(busSizeList),busSizeList[0][1],2),np.nan) # R3 matrix of bus x maxSize x lat/lon
	busTimestamps = dict() #dict of lists of timestamps of each entry of a certain bus indexed by bus_id	
	with database.cursor() as cursor:
		cursor.execute(""" 
		SELECT 
			bus_id,time_detection,latitude,longitude,line_reported 
		FROM bus_data 
		WHERE 
			bus_id IN %s 
			AND time_detection BETWEEN %s AND %s  
		ORDER BY bus_id, time_detection """, (
			tuple([i[0] for i in busSizeList]),
			desiredDate,nextDate
			)
		)
		busTable = pd.DataFrame(cursor.fetchall(),columns=['bus_id','time_detection','lat','lon','reported'])
	
	log.debug("Sorting requested bus data")
	busTable = busTable.sort_values(["bus_id","time_detection"]).copy()

	log.debug("Creating bus matrix")
	for index,busInfo in enumerate(busSizeList):
		busId,busSize = busInfo
		#busTimestamps[busId] = list(busTable.loc[busTable['bus_id'] == busId]['time_detection'])
		busTrajectoryArray = np.array(busTable.loc[busTable["bus_id"] == busId][["lat","lon"]])
		busMatrix[index] = np.pad(busTrajectoryArray,((0,busSizeList[0][1] - busSize),(0,0)),constant_values=np.nan)

	log.debug("Requesting line data from database")
	lineMatrix = np.full((len(lineSizeList),lineSizeList[0][2],2),np.nan)
	with database.cursor() as cursor:
		cursor.execute("""
			SELECT 
				line_id,direction,latitude,longitude
			FROM line_data
			WHERE 
				line_id IN %s 
			ORDER BY position;
			""",(
				tuple([i[0] for i in lineSizeList])
			,)
		)
		lineTable = pd.DataFrame(cursor.fetchall(),columns=['line_id','direction','lat','lon'])

	log.debug("creating line matrix")
	for index,lineInfo in enumerate(lineSizeList):
		lineId, lineDirection, lineSize = lineInfo
		lineTrajectoryArray = np.array(lineTable.loc[(lineTable["line_id"] == lineId) & (lineTable['direction'] == str(lineDirection))][["lat","lon"]])
		lineMatrix[index] = np.pad(lineTrajectoryArray,((0,lineSizeList[0][2] - lineSize),(0,0)),constant_values=np.nan)
 

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
	#performanceData['extrapolated-maximum-d-size-mbytes'] = (32*busMatrix.shape[0]*busMatrix.shape[1]*lineMatrix.shape[0]*lineMatrix.shape[1])/(8*10**6)
	#performanceData['maximum-d-size-batched-mbytes'] = (32*int(CONFIGS['lineDetection']['busStepSize'])*busMatrix.shape[1]*int(CONFIGS['lineDetection']['lineStepSize'])*lineMatrix.shape[1])/(8*10**6)
	#performanceData['iterations-amount'] = (performanceData['bus-amount']/int(CONFIGS['lineDetection']['busStepSize']))*(performanceData['line-amount']/int(CONFIGS['lineDetection']['lineStepSize']))
	
	# Saving temporary data in case of crash for fast recovery (not implemented yet TODO)
	with open("/tmp/busSizeList.pickle",'wb') as fil:
		pickle.dump(busSizeList,fil)

	with open("/tmp/lineSizeList.pickle",'wb') as fil:
		pickle.dump(lineSizeList,fil)

	with open("/tmp/busMatrix.pickle",'wb') as fil:
		pickle.dump(busMatrix,fil)

	with open("/tmp/lineMatrix.pickle",'wb') as fil:
		pickle.dump(lineMatrix,fil)

	with open("/tmp/busTimestamps.pickle",'wb') as fil:
		pickle.dump(busTimestamps,fil)
	
	log.debug("Starting correction process")

	# -------------------
	# Line detection and correction phase
	# -------------------
	# Line detection phase
	mes.start("line-detection")

	mes.start("line-detection-function")
	detectionMatrix = LineDetection.FilterData(busMatrix,lineMatrix,busSizeList,lineSizeList,CONFIGS,log)
	mes.end("line-detection-function")
	
	mes.start('line-detection-saving')
	with open("/tmp/detectMatrix.pickle.tmp",'wb') as fil:
		pickle.dump(detectionMatrix,fil)
	mes.end("line-detection-saving")
	
	mes.end("line-detection")


	# Line correction phase
	mes.start("line-correction")

	mes.start('line-correction-function')
	busResultTable = LineCorrection.CorrectData((detectionMatrix > float(CONFIGS['default_correction_method']['detectionPercentage'])),busMatrix,lineMatrix,busSizeList,lineSizeList,CONFIGS) # Matriz R^3 de (entidade x indice)
	mes.end("line-correction-function")

	mes.start('line-correction-saving')
	with open("/tmp/busResultTable.pickle.tmp",'wb') as fil:
		pickle.dump(busResultTable,fil)
	mes.end("line-correction-saving")
	
	mes.end("line-correction")
	log.debug("correction process done")
	# ----------------------------------------------------------------------------
	# Database saving
	# -----------------------------------------------------------------------------
	mes.start('database-insertion')
	if busResultTable.shape[0] == 0:
		print("Done: NO MATCHES WERE DETECTED")
		exit(0)
	
	log.debug("Sending data to database")
	# Inserting detected values into corrected places
	busTable["line_corrected"] = ""
	print(busTable[['line_corrected']] == None)
	for index in busResultTable.columns.to_list():
		busSize = [i[1] for i in busSizeList if i[0] == index][0]
		busTable.loc[busTable['bus_id'] == index,"line_corrected"] = busResultTable[index].to_list()[:busSize]

	busTable.loc[busTable['line_corrected'] == "","line_corrected"] = "No_line"

	#busTable[['time_detection']] = busTable[['time_detection']].astype("int64") // 10**9
	busTable['bus_id'] = busTable.bus_id.apply(lambda x: x[1:-1] if x[0] == "'" else x)
	busTable['reported'] = busTable.reported.apply(lambda x: x[1:-1] if x[0] == "'" else x)
	

	buff = io.StringIO()
	# inserting new ones in bus_data table
	busTable = busTable.rename(columns={"lat":"latitude","lon":"longitude","reported":"line_reported","line_corrected":"line_detected"})

	busTable.to_csv(buff,header=False,index=False,columns=("time_detection","bus_id","latitude","longitude","line_reported","line_detected"))
	buff.seek(0)
	with database.cursor() as cursor:
		cursor.copy_from(buff,"bus_data",sep=",")#,columns=("bus_id","time_detection","latitude","longitude","line_reported","line_detected"))
	database.commit()
	
	# Deleting original values from bus_data (identified by none in line_detected column)
	with database.cursor() as cursor:
		cursor.execute(
			"""
			DELETE FROM bus_data
			WHERE
				time_detection BETWEEN %s AND %s
				AND bus_id IN %s AND
				line_detected = ''
			""",
			(
				desiredDate,nextDate,
				tuple([i[0] for i in busSizeList])
			)
		)
	database.commit()

   
	mes.end('database-insertion')
	
	
if __name__ == '__main__':
	main()
	
