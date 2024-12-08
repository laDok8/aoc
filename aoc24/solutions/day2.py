from utils import scrape

def aoc2():
    inp = scrape()
    correct = 0
    for line in inp:
        diff_line = []
        nums = list(map(int, line.split()))
        for i in range(len(nums)-1):
            diff_line.append(nums[i] - nums[i+1])

        diffs_uniq = set(diff_line)
        if all(3 >= abs(x) >= 1 for x in diffs_uniq):
            if all(x > 0 for x in diff_line) or all(x < 0 for x in diff_line):
                correct += 1
    print("part 1:", correct)

    correct2 = 0
    for line in inp:
        nums = list(map(int, line.split()))
        is_safe = False

        for i in range(len(nums)):
            temp_nums = nums[:i] + nums[i+1:]
            diff_line = [temp_nums[j] - temp_nums[j+1] for j in range(len(temp_nums)-1)]
            diffs_uniq = set(diff_line)
            if all(3 >= abs(x) >= 1 for x in diffs_uniq):
                if all(x > 0 for x in diff_line) or all(x < 0 for x in diff_line):
                    is_safe = True
                    break

        if is_safe:
            correct2 += 1

    print("part 2:", correct2)
