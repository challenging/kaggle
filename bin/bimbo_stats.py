#!/usr/bin/env python

import sys
import click

from utils import log, INFO
from bimbo.stats_benastalk import producer, consumer

@click.command()
@click.option("--is-testing", is_flag=True, help="testing mode")
@click.option("--column", default="agency_id", help="agency_id|channel_id|route_id|product_id")
@click.option("--mode", required=True, help="producer mode | consumer mode")
def stats(is_testing, column, mode):
    if mode == "producer":
        producer(column, is_testing, ttr=86400)
    elif mode == "consumer":
        consumer()
    else:
        log("Not implement this mode({})".format(mode), INFO)

        sys.exit(999)

if __name__ == "__main__":
    stats()
