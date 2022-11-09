FROM python:3.8
#Set working directory to app
COPY . /app
WORKDIR /app
#Install requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8000
CMD python ./app.py --ip 0.0.0.0