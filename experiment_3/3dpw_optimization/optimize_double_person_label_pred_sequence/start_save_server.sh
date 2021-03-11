#!/bin/bash

debug='--debug' # debug on
#debug='' # debug off

python save_server.py --port 40034 $debug &
python save_server.py --port 40035 $debug &
python save_server.py --port 40036 $debug &
python save_server.py --port 40037 $debug &


