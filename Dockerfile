#
# Correction module's Dockerfile
# Module's description: 
#   This module is for correcting bus trajectory data from the FAS-Bus project Rio's implementation.
#
# Dockerfile stages:
#   environment - to mount to host's app file for continuos modification without the need to rebuild
#   test - Container that creates a full environment for testing, executes a test script and exits, indicating the test results on the error code.
#   production - Final build for usage

#   More info at https:/fasbus.gta.ufrj.br/


#
# Base container
#
FROM cupy/cupy:latest AS base

# installing essencial libraries
RUN apt update && apt install -y libpq-dev && python3 -m pip install numpy pandas psycopg2 

#
# environment container
# Use this for continuos development of the python code
#
FROM base AS environment

RUN mkdir /app /filters

VOLUME [ "/app" , "/filters" ]

WORKDIR /app

CMD ["/bin/bash"]


#
# test container
# Executes tests scripts and installs aditional libraries
# 
FROM base as test

COPY ./app/ /app/

COPY ./tests/ /tests/

VOLUME ["/filters"]

WORKDIR /tests

CMD ["python3","main_test.py"]

#
# Production container
# Use for final project deployment
#
FROM base as production

RUN mkdir /filters

VOLUME ["/filters"]

COPY ./app/ /app/

WORKDIR /app

