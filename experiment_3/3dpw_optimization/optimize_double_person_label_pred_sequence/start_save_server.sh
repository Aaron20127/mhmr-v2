#!/bin/bash

debug='--debug' # debug on
#debug='' # debug off

python save_server.py --port 60030 $debug &
python save_server.py --port 60031 $debug &
python save_server.py --port 60032 $debug &
python save_server.py --port 60033 $debug &
python save_server.py --port 60034 $debug &
python save_server.py --port 60035 $debug &
python save_server.py --port 60036 $debug &
python save_server.py --port 60037 $debug &
python save_server.py --port 60038 $debug &
python save_server.py --port 60039 $debug &
