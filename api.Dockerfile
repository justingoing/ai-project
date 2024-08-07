FROM python:3.8.8-buster


COPY ./api ./api
COPY ./data ./data
WORKDIR /api
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN chmod u+x api.py

ENTRYPOINT [ "python", "api.py" ]

EXPOSE 5000