FROM python:3.7-slim-bullseye
LABEL maintainer="@satake"

#
# install packages
#
RUN apt update && apt install -y \
    vim \
    build-essential \
    libsndfile1 \
    libasound2-dev \
    libjack-dev \
    portaudio19-dev \
    ca-certificates \
    curl \
    unzip \
    git && \
    rm -rf /var/lib/apt/lists/*

#
# install & setup megenta
#
RUN pip install magenta
RUN git clone https://github.com/tensorflow/magenta.git /opt/magenta && \
    cd /opt/magenta && \
    pip install -e .

#
# setup magenta.sh
#
RUN curl https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip > /opt/maestro_checkpoint.zip
RUN  cd /opt && unzip /opt/maestro_checkpoint.zip

# COPY wavs/ /opt/wavs
COPY onsets_frames_transcription_transcribe.py /opt/magenta/magenta/models/onsets_frames_transcription/
COPY file_list.txt /opt/file_list.txt

COPY transcribe.sh /opt/transcribe.sh

RUN chmod +x /opt/transcribe.sh
CMD ["/opt/transcribe.sh"]
