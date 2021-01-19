#!/bin/bash

#debug='--debug' # debug on
debug='' # debug off

python save_server.py --port 6000 $debug &
python save_server.py --port 6001 $debug &
python save_server.py --port 6002 $debug &
python save_server.py --port 6003 $debug &
python save_server.py --port 6004 $debug &
python save_server.py --port 6005 $debug &
python save_server.py --port 6006 $debug &
python save_server.py --port 6007 $debug &
python save_server.py --port 6008 $debug &
python save_server.py --port 6009 $debug &
