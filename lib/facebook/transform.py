#!/usr/bin/env python

import sys

if __name__ == "__main__":
    with open(sys.argv[1], "rb") as INPUT:
        pre_row_id = None
        pre_line = None

        for line in INPUT:
            row_id = line.split(",")[0]

            if pre_row_id != None and row_id != pre_row_id:
                print pre_line.strip()

            pre_row_id = row_id
            pre_line = line

        print pre_line.strip()
