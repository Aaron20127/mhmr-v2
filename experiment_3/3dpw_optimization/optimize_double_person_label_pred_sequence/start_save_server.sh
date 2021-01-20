#!/bin/bash

debug='--debug' # debug on
#debug='' # debug off

python save_server.py --port 60020 $debug &
python save_server.py --port 60021 $debug &
python save_server.py --port 60022 $debug &
python save_server.py --port 60023 $debug &
python save_server.py --port 60024 $debug &
python save_server.py --port 60025 $debug &
python save_server.py --port 60026 $debug &
python save_server.py --port 60027 $debug &
python save_server.py --port 60028 $debug &
python save_server.py --port 60029 $debug &
