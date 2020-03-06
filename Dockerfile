FROM tensorflow/tensorflow:1.11.0-gpu-py3

WORKDIR /backgammon

COPY backgammon /backgammon/backgammon
