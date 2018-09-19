#!/bin/sh

if [ -f ".mpmtrainid" ]; then
    if [ -d /proc/`cat .mpmtrainid` ]; then
      echo stop `cat .mpmtrainid` && kill `cat .mpmtrainid` && rm .mpmtrainid
  else
     echo remove dead pid `cat .mpmtrainid` && rm .mpmtrainid
  fi
else
    echo nothing to stop, can not find .mpmtrainid file
fi