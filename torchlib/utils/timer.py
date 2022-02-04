import datetime
import time


class Timer(object):
    def __init__(self):
        self.start_time = None

    def reset(self):
        self.start_time = time.time()

    def get_time_elapsed(self):
        return datetime.timedelta(seconds=int(time.time() - self.start_time))
    