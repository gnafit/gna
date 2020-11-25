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
        print(textwrap.dedent(cls.__doc__))
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

def print_tldr(command, arg, tldr):
    if not tldr:
        return
    tldr = textwrap.dedent(tldr)
    print('\033[32mTLDR\033[0m for {} {}\n'.format(command, arg))
    print(tldr)
    print()

help.__tldr__ = {
            "" : """\
                 \033[32mRetrieve the description and examples of the comment/help UI commands with:
                 \033[31m./gna -- help comment
                 ./gna -- help help\033[0m

                 \fggreenIf an UI command provides extra tldr for a longer substring, it will be printed. Try:
                 \033[31m./gna -- help help help\033[0m
                 """,
            "help" : """\
                     \033[32mThis is an example tldr, printed for 'help help' version:
                     \033[31m./gna -- help help help\033[0m
                     """
            }
