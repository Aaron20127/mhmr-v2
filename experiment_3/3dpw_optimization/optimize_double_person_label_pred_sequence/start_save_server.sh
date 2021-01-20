#!/bin/bash

debug='--debug' # debug on
#debug='' # debug off

python save_server.py --port 60000 $debug &
python save_server.py --port 60001 $debug &
python save_server.py --port 60002 $debug &
python save_server.py --port 60003 $debug &
python save_server.py --port 60004 $debug &
python save_server.py --port 60005 $debug &
python save_server.py --port 60006 $debug &
python save_server.py --port 60007 $debug &
python save_server.py --port 60008 $debug &
python save_server.py --port 60009 $debug &
