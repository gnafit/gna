

printlevel = 0
singlemargin = '    '
marginflag = False

class nextlevel():
    def __enter__(self):
        global printlevel
        printlevel+=1

    def __exit__(self, *args, **kwargs):
        global printlevel
        printlevel-=1

def current_level():
    return printlevel

def printmargin(kwargs):
    global marginflag
    prefix = kwargs.pop('prefix', None)
    postfix = kwargs.pop('postfix', None)
    prefixopts = kwargs.pop('prefixopts', dict(end=''))
    postfixopts = kwargs.pop('postfixopts', dict(end=' '))
    if marginflag:
        return

    if prefix:
        print(*prefix, **prefixopts)

    print(singlemargin*printlevel, sep='', end='')

    if postfix:
        print(*postfix, **postfixopts)

    marginflag=True

def resetmarginflag(*args, **kwargs):
    global marginflag

    for arg in args+(kwargs.pop('sep', ''), kwargs.pop('end', '\n')):
        if not isinstance(arg, str):
            arg = str(arg)
        if '\n' in arg:
            marginflag=False
            return

def printl(*args, **kwargs):
    printmargin(kwargs)
    print(*args, **kwargs)
    resetmarginflag(*args, **kwargs)
