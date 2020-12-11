
class AverageLoss():
    def __init__(self):
        self.has_init = False
        self.count = 0

    def add(self, dict_in):
        if not self.has_init:
            self.has_init = True
            self.total_dict = {}
            for k, v in dict_in.items():
                self.total_dict[k] = v

        else:
            for k, v in dict_in.items():
                self.total_dict[k] += v

        self.count += 1

    def get_average(self):
        average = {}

        for k, v in self.total_dict.items():
            average[k] = v / self.count

        return average
