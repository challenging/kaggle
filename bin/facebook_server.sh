#!/bin/sh

PATH_MONGODB=~/Documents/mongodb/facebook_competition
PATH_BEANSTALK=~/Documents/beanstalk/facebook_competition

action=
while getopts "s:" opt; do
  case $opt in
    s)
      action=$OPTARG
      ;;
  esac
done

if [ ! -d ${PATH_MONGODB} ]; then
    mkdir -p ${PATH_MONGODB}
fi

if [ ! -d ${PATH_BEANSTALK} ]; then
    mkdir -p ${PATH_BEANSTALK}
fi

if [ "${action}" == "start" ]; then
    # startup the server of Mongodb
    mongod --dbpath ${PATH_MONGODB} -nssize 64 &

    # startup the server of beanstalk
    IP=$(ifconfig | grep inet | grep 192 | awk '{print $2}')
    beanstalkd -b ${PATH_BEANSTALK} -l ${IP} -z 655360 &
elif [ "${action}" == "stop" ]; then
    pid_mongo=$(ps -ef | grep mongo | grep -v grep | awk '{print $2}')
    kill ${pid_mongo}

    rc=$?
    if [ ${rc} -ne 0 ]; then
        echo "kill the process of mongod service"
    else
        echo "fail in killing the process(${pid_mongo}) of mongod service(${rc})"
    fi

    pid_beanstalk=$(ps -ef | grep beanstalkd | grep -v grep | awk '{print $2}')
    kill ${pid_beanstalk}

    rc=$?
    if [ ${rc} -ne 0 ]; then
        echo "kill the process of beanstalkd service"
    else
        echo "fail in killing the process(${pid_beanstalk}) of beanstalk service(${rc})"
    fi
else
    echo "wrong action - ${action}"
fi
