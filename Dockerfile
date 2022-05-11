FROM ubuntu:focal
RUN apt-get update && apt-get install -y sudo software-properties-common g++
COPY . /tomcat-tmm
WORKDIR /tomcat-tmm
RUN ./tools/install
