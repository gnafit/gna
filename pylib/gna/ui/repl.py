from gna.ui import basecmd

class cmd(basecmd):
    def run(self):
        import IPython
        IPython.embed()
