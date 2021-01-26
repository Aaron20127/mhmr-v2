#!/bin/bash

debug='--debug' # debug on
#debug='' # debug off

python save_server.py --port 40030 $debug &
python save_server.py --port 40031 $debug &
python save_server.py --port 40032 $debug &
python save_server.py --port 40033 $debug &


