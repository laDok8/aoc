import regex as re

from utils import scrape


def aoc3():
    sum1, sum2 = 0, 0
    mul_regex = r'mul\((\d*),(\d*)\)'
    all_text = ''.join(scrape())
    mul_re = re.compile(mul_regex)

    for call in mul_re.findall(all_text):
        sum1 += int(call[0]) * int(call[1])
    print('part 1:', sum1)

    # part 2
    do_data = re.split(r"don't\(\).*?(?:do\(\)|$)", all_text)
    for call in mul_re.findall(''.join(do_data)):
        sum2 += int(call[0]) * int(call[1])
    print('part 2:', sum2)
    # part 2: 102631226
