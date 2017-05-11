FROM continuumio/anaconda3:4.3.0
MAINTAINER Ibraheem AJ "ibraheemaj@icloud.com"

EXPOSE 5000

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN apt-get install libglib2.0-0 
RUN apt-get -y install libgl1-mesa-glx
RUN conda install -c rdkit rdkit -y
RUN conda install scikit-learn -y
RUN conda install -c anaconda flask=0.12.1 -y
RUN pip install graphviz

COPY . /bbb
WORKDIR /bbb


CMD ["python","server.py","--model=brain_2017-05-01_bbb.pkl"]