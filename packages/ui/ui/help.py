from gna.ui import basecmd
from argparse import REMAINDER
from gna.dispatch import getmodules, loadcmdclass
import textwrap

class help(basecmd):
    """Print help on a command

    Similar to the "--help" UI option, but is not executed immediately.
    """
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('command', help='command to print help on')
        parser.add_argument('args', nargs=REMAINDER, help="ignored arguments")

    def init(self):
        name = self.opts.command.replace('-', '_')
        modules = getmodules()

        if name not in modules:
            print("UI module '{}' not found".format(name))
            return

        cls, _ = loadcmdclass(modules, name)

        if not cls.__doc__:
            print("UI module '{}' provides no docstring".format(name))
            return

        print("\033[32mUI module '{}' docstring:\033[0m\n".format(name))
        print(cls.__doc__)
        print()

        tldr = getattr(cls, '__tldr__', None)
        if isinstance(tldr, str):
            print_tldr(name, '', tldr)
        elif isinstance(tldr, dict):
            print_tldr(name, '', tldr.get(''))
            trystr = None
            for arg in tuple(self.opts.args):
                if trystr:
                    trystr = ' '.join((trystr, arg))
                else:
                    trystr = arg

                print_tldr(name, trystr, tldr.get(trystr))


    __tldr__ = {
            "" : """\
                Retrieve the description and examples of the comment/help UI commands with:
                  ./gna -- help comment
                  ./gna -- help help

                If an UI command provides extra tldr for a longer substring, it will be printed. Try:
                  ./gna -- help help help
            """,
            "help" : """This is an example tldr, printed for 'help help' version:
                  ./gna -- help help help
            """
            }

def print_tldr(command, arg, tldr):
    if not tldr:
        return
    tldr = textwrap.dedent(tldr)
    print('\033[32mTLDR\033[0m for {} {}\n'.format(command, arg))
    print(tldr)
    print()
