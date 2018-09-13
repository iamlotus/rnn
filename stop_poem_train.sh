#!/bin/sh

if [ -f ".poemtrainpid" ]; then
    if [ -d /proc/`cat .poemtrainpid` ]; then
      echo stop `cat .poemtrainpid` && kill `cat .poemtrainpid` && rm .poemtrainpid
  else
     echo remove dead pid `cat .poemtrainpid` && rm .poemtrainpid
  fi
else
    echo nothing to stop, can not find .poemtrainpid file
fi