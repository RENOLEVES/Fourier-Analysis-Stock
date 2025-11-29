from backtesting import *
from sampling import *


class Modeller(Sampling, Backtesting):

    def run(self, style='basic', scoring=None):
        if style == 'basic':
            self.run_basic(scoring)

    def run_basic(self, scoring=None):
        self.preprocessing()
        self.rf_manual(scoring=scoring, complexity='basic')
        self.output()