#!/bin/sh

BASEPATH=$(dirname $0)
BEANSTALK_CLIENT=${BASEPATH}/bimbo_stats.py

N_NUM=$1

for idx in $(seq 1 ${N_NUM});
do
    ${BEANSTALK_CLIENT} --mode consumer &
done
