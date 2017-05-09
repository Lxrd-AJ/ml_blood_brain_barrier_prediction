FROM continuumio/anaconda3
MAINTAINER Ibraheem AJ "ibraheemaj@icloud.com"

EXPOSE 5000

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
# RUN apt-get install python-rdkit librdkit1 rdkit-data
RUN conda install -c rdkit rdkit -y
RUN conda install scikit-learn -y
RUN conda install -c anaconda flask=0.12.1 -y
RUN pip install graphviz
#RUN pip install Flask

COPY . /bbb
WORKDIR /bbb

# python server.py --model=brain_2017-05-01_bbb.pkl 
CMD ["python","server.py","--model=brain_2017--05-01_bbb.pkl"]