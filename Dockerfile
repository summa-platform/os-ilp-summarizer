FROM ubuntu:bionic

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt-get install -y build-essential python3-distutils cython3 python3-nltk python3-flask python3-requests python3-tornado python3-lxml

RUN python3 -c "import nltk;nltk.download('punkt')"

WORKDIR /root
ENV baseDir .
ADD ${baseDir}/AD3  /root/AD3
WORKDIR /root/AD3
RUN python3 /root/AD3/setup.py install

ADD ${baseDir}/summarizer  /root/summarizer
RUN ln -s ~/AD3/build/lib.linux-x86_64-3.6/ad3/simple_inference.py /root/summarizer/
RUN ln -s ~/AD3/build/lib.linux-x86_64-3.6/ad3/factor_graph.cpython-36m-x86_64-linux-gnu.so  /root/summarizer/

EXPOSE 5000

WORKDIR /root/summarizer
CMD /root/summarizer/app.py