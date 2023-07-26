import math


class RunningStats:
    """
    Computes the running mean and standard deviation of a sequence of numbers.
    Based on http://www.johndcook.com/standard_deviation.html
    """

    __slot__ = ['n', 'old_m', 'new_m', 'old_s', 'new_s']

    def __init__(self, n=0, old_m=0, new_m=0, old_s=0, new_s=0):
        self.n = n
        self.old_m = old_m
        self.new_m = new_m
        self.old_s = old_s
        self.new_s = new_s

    def copy(self):
        return RunningStats(self.n, self.old_m, self.new_m, self.old_s, self.new_s)

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())
