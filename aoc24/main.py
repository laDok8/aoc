#!/usr/bin/env python
import datetime

import solutions as s

if __name__ == '__main__':
    # start aoc for given calendar day
    today = datetime.date.today().day
    aocs = [s.aoc1, s.aoc2, s.aoc3]
    aocs[today - 1]()
