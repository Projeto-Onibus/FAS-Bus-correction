import sys
import logging

def logger(verbosityLevel):
	logger = logging.getLogger(sys.argv[0])

	logger.setLevel(logging.DEBUG)
# create console handler and set level to debug
	ch = logging.StreamHandler()

	logLevel = logging.ERROR * (verbosityLevel == 0) + \
			logging.WARNING * (verbosityLevel == 1) + \
			logging.INFO * (verbosityLevel == 2) + \
			logging.DEBUG * (verbosityLevel >= 3)

	ch.setLevel(logLevel)

# create formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
	ch.setFormatter(formatter)

# add ch to logger
	logger.addHandler(ch)

	return logger
