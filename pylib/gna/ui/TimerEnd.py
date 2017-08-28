import time
from gna.ui import basecmd
import ROOT
from gna.env import env



class cmd(basecmd):
    def run(self):
        endtime = time.time()
        print "Time of run: ", endtime - env.starttime
