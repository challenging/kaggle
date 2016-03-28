#!/usr/bin/env python

import os
import click

# Refer from: http://click.pocoo.org/5/commands/
plugin_folder = os.path.join(os.path.dirname(__file__), 'bin')

class KaggleCli(click.MultiCommand):

    def list_commands(self, ctx):
        cmds = []
        for filename in os.listdir(plugin_folder):
            if filename.endswith('.py') and not filename.startswith('_'):
                cmds.append(filename[:-3])
        return sorted(cmds)

    def get_command(self, ctx, name):
        ns = {}
        fn = os.path.join(plugin_folder, name + '.py')
        with open(fn) as f:
            code = compile(f.read(), fn, 'exec')
            eval(code, ns, ns)
        return ns['cli']

@click.group(cls=KaggleCli, help='For the Kaggle Competition')
def cli():
    pass

if __name__ == '__main__':
    cli()
