from gna.env import env
import time
from gna.ui import basecmd
import ROOT
import gna


class cmd(basecmd):
    def run(self):
        env.starttime = time.time()
