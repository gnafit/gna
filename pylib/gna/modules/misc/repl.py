from gna.ui import basecmd

class repl(basecmd):
    def run(self):
        import IPython
        IPython.embed()
