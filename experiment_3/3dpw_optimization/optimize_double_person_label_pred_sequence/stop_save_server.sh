#!/bin/bash

kill -15 `ps -aux | grep save_server | awk -F" " '{print $2}'`