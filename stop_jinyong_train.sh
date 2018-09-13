#!/bin/sh

if [ -f ".jytrainpid" ]; then
    if [ -d /proc/`cat .jytrainpid` ]; then
      echo stop `cat .jytrainpid` && kill `cat .jytrainpid` && rm .jytrainpid
  else
     echo remove dead pid `cat .jytrainpid` && rm .jytrainpid
  fi
else
    echo nothing to stop, can not find .jytrainpid file
fi