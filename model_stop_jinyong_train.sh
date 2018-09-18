#!/bin/sh

if [ -f ".mjytrainpid" ]; then
    if [ -d /proc/`cat .mjytrainpid` ]; then
      echo stop `cat .mjytrainpid` && kill `cat .mjytrainpid` && rm .mjytrainpid
  else
     echo remove dead pid `cat .mjytrainpid` && rm .mjytrainpid
  fi
else
    echo nothing to stop, can not find .mjytrainpid file
fi