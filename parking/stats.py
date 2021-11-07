import math

class Results:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def eval(self, truth, val):
        if truth:
            if val:
                self.tp += 1
            else:
                self.fn += 1
        else:
            if val:
                self.fp += 1
            else:
                self.tn += 1

    def get_accuracy(self):
        top = float(self.tp + self.tn)
        bottom = float(self.tp + self.tn + self.fp + self.fn)
        return top / bottom

    def get_f1(self):
        precision = float(self.tp) / float(self.tp + self.fp)
        recall = float(self.tp) / float(self.fn + self.tn)
        return 2.0 * float(precision * recall) / float(precision + recall)

    def get_mcc(self):
        top = float(self.tp * self.tn - self.fp + self.fn)
        bottom = float((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))
        return top / math.sqrt(bottom)



def get_F1_score(results):
    precision = float(results["tp"]) / float(results["tp"] + results["fp"])
    recall = float(results["tp"]) / float(results["fn"] + results["tn"])

    return 2.0 * float(precision * recall) / float(precision + recall)

def get_MCC(results):
    top = float(results["tp"] * results["tn"] - results["fp"] + results["fn"])
    bottom = float((results["tp"] + results["fp"]) * (results["tp"] + result["fn"]) * (results["tn"] + results["fp"]) * (results["tn"] + results["fn"]))
    return top / math.sqrt(bottom)