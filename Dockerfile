FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
COPY . /home
EXPOSE 8501
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    #libopencv-dev \
    #python3-opencv \
    ffmpeg \
    libsm6 \
    libxext6
#RUN pip install --upgrade numpy pandas
RUN pip install -r /home/requirements.txt
WORKDIR /home
CMD [ "streamlit", "run", "app.py" ]