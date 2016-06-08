#!/bin/sh

BASEPATH=.
BASEPATH_LIB=${BASEPATH}/../lib/facebook

BEANSTALK_CLIENT=${BASEPATH_LIB}/facebook_beanstalk.py

N_NUM=$1

for idx in $(seq 1 ${N_NUM});
do
    ${BEANSTALK_CLIENT} &
done
