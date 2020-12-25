import ROOT as R

class TFileContext(object):
    """A context for opening a ROOT file"""
    def __init__(self, filename, mode='read'):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = R.TFile(self.filename, self.mode)
        if self.file.IsZombie():
            raise Exception('Unable to read ({}) ROOT file: {}'.format(self.mode, self.filename))
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.Close()
