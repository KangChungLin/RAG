FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY . /app

ADD ./requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install transformers==4.34.1

RUN pip install sentence-transformers==2.2.2 --extra-index-url https://download.pytorch.org/whl/cpu