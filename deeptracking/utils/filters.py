import collections


class MeanFilter:
    def __init__(self, length):
        self.value_list = collections.deque(maxlen=length)

    def compute_mean(self, value):
        self.value_list.append(value)
        mean = sum(self.value_list) / len(self.value_list)
        return mean

    def normalize(self, quat):
        lengthD = 1.0 / (quat[3] * quat[3] + quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2])
        return quat * lengthD

    def is_close(self, q1, q2):
        dist = q1.dot(q2)
        return dist >= 0




