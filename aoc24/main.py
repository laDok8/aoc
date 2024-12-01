#!/usr/bin/env python
import datetime


from utils import scrape
from solutions import *

if __name__ == '__main__':
    # start aoc for given calendar day
    today = datetime.date.today().day
    aocs = [aoc1]
    aocs[today - 1]()
