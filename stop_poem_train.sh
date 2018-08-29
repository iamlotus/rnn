#!/bin/sh

if [ -f ".trainpid" ]; then
    if [ -d /proc/`cat .trainpid` ]; then
      echo stop `cat .trainpid` && kill `cat .trainpid` && rm .trainpid
  else
     echo remove dead pid `cat .trainpid` && rm .trainpid
  fi
else
    echo nothing to stop, can not find .trainpid file
fi