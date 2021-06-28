FROM cupy/cupy:latest

RUN apt update && apt install -y libpq-dev && python3 -m pip install numpy pandas psycopg2

COPY ./app/ /app/

WORKDIR /app