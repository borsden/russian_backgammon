FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /backgammon

COPY backgammon /backgammon/backgammon
COPY play.py /backgammon/play.py

CMD python3 play.py