#!/usr/bin/env python

from sys import stdout, stderr
import itertools

isatty = stdout.isatty()

def statusline(line):
    """Print status line if condition is true. Correctly clears the line before printing"""
    stderr.write( '\r\033[2K%s\r'%line )
    stderr.flush()
##end def statusline

def human_time( left ):
    left*=1000.
    lleft, lunit = left, 'ms'
    for div, unit in ( ( 1000., 's' ), ( 60., 'm' ), ( 60., 'h' ),
                       ( 24., 'days' ), ( 7., 'weeks' ),  ( 30./7.0, 'months' ),
                       ( 365./30., 'years' ), ( 100., 'centuries' ), ( 10., 'millenia' ) ):
        left/=div
        if left<1.0:
            break

        lleft, lunit = left, unit

    return lleft, lunit

progress_it = itertools.cycle( '>})|({<{(|)}' )
def progress( message, done, total, time_elapsed=None ):
    global progress_it
    if done==total:
        statusline( '' )
        return

    width = 40
    r_done = done/total
    w_done = int(r_done*width)
    w_remains = width-w_done-1
    prg = '='*w_done+progress_it.next()+' '*w_remains
    lines = [ '%s: [%s]%3i%% (%i/%i)'%( message, prg, r_done*100., done, total ) ]
    if not time_elapsed is None:
        time_it, unit_it = human_time( time_elapsed/done )
        time_el, unit_el = human_time( time_elapsed )
        time_r, unit_r = human_time( (total-done)*time_elapsed/done )
        lines.append( ', it/elapsed: %.0f %s/%.2f %s, remains: %.2f %s'%(
            time_it, unit_it, time_el, unit_el, time_r, unit_r
        ) )
    statusline( ''.join( lines ) )
##end def statusline

commands = dict(
    reset = '0',
    bright = '1',
    dim = '2',
    underscore = '4',
    blink = '5',
    reverse = '7',
    hidden = '8',
    black = '30',
    red = '31',
    green = '32',
    yellow = '33',
    blue = '34',
    magenta = '35',
    cyan = '36',
    gray = '37',
    bblack = '40',
    bred = '41',
    bgreen = '42',
    byellow = '43',
    bblue = '44',
    bmagenta = '45',
    bcyan = '46',
    bgray = '47'
)

style = dict( error='red', warning='magenta', good='green', bad='red', normal='reset',
              attention=[ 'blink', 'bright', 'red' ], debug='gray', info='yellow' )

def get_command( opts ):
    if opts in style.keys():
        opts = style[opts]
    if type(opts)==str:
        opts = [ opts ]
    return '\033[%sm'%( ';'.join( ( commands[c] for c in opts ) ) )
##end def get_command

def colored( text, opts ):
    return '%s%s\033[0m'%( get_command(opts), text )
##end def colored

alread_printed_messages=[]
def print_colored( opts, *args, **kwargs ):
    if kwargs.pop( 'once', False ):
        if args in alread_printed_messages:
            return
        alread_printed_messages.append( args )
    if not isatty:
        print( *args, **kwargs )
        return

    if opts:
        stdout.write( get_command( opts ) )
    print( *args, **kwargs )
    stdout.write('\033[0m')
    stdout.flush()
##end def print_colored
fprint = print_colored

def pprint_color(obj):
    """Pretty printer with syntax highlight"""
    import pprint
    try:
        from pygments import highlight
        from pygments.lexers import PythonLexer
        from pygments.formatters import Terminal256Formatter
        print(highlight(pprint.pformat(obj), PythonLexer(), Terminal256Formatter(style='native')))
    except ImportError:
        pprint.pprint( obj )

def set_style( opts ):
    stdout.write( get_command( opts ) )

redirected = dict()
def redirect_output( fname, stdout=True, stderr=False, cstdout=False ):
    print_colored( 'info', 'Redirecting:', end=' ' )
    if stderr:
        print_colored( 'info', '[%s]'%fname.replace( '.out', '.err' ), end=' ' )
    if stdout:
        print_colored( 'info', '[%s]'%fname, end=' ' )
    if cstdout:
        print_colored( 'info', '[%s]'%fname.replace( '.out', '.cout' ), end=' ' )
    print()

    import sys
    if stderr:
        redirected['stderr'] = sys.stderr
        sys.stderr = open( fname.replace( '.out', '.err' ), 'w' )
    if stdout:
        redirected['stdout'] = sys.stdout
        sys.stdout = open( fname, 'w' )
    if cstdout:
        import ROOT
        redirected['redir'] = redir = ROOT.OutputRedir()
        redir.redir_stdout( fname.replace( '.out', '.cout' ) )
