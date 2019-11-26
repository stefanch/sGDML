#!/usr/bin/python

# MIT License
#
# Copyright (c) 2018 Stefan Chmiela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function

import argparse
import os
import re
import sys

from . import __version__
from .utils import ui

if sys.version[0] == '3':
    raw_input = input


try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen


def download(command, file_name):

    base_url = 'http://www.quantum-machine.org/gdml/' + (
        'data/npz/' if command == 'dataset' else 'models/'
    )
    request = urlopen(base_url + file_name)
    file = open(file_name, 'wb')
    filesize = int(request.headers['Content-Length'])

    size = 0
    block_sz = 1024
    while True:
        buffer = request.read(block_sz)
        if not buffer:
            break
        size += len(buffer)
        file.write(buffer)

        ui.progr_bar(
            size,
            filesize,
            disp_str='Downloading: {}'.format(file_name),
            sec_disp_str='{:,} bytes'.format(filesize),
        )
    file.close()


def main():

    base_url = 'http://www.quantum-machine.org/gdml/'

    parser = argparse.ArgumentParser()

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '-o',
        '--overwrite',
        dest='overwrite',
        action='store_true',
        help='overwrite existing files',
    )

    subparsers = parser.add_subparsers(title='commands', dest='command')
    subparsers.required = True
    parser_dataset = subparsers.add_parser(
        'dataset', help='download benchmark dataset', parents=[parent_parser]
    )
    parser_model = subparsers.add_parser(
        'model', help='download pre-trained model', parents=[parent_parser]
    )

    for subparser in [parser_dataset, parser_model]:
        subparser.add_argument(
            'name',
            metavar='<name>',
            type=str,
            help='item name',
            nargs='?',
            default=None,
        )

    args = parser.parse_args()

    if args.name is not None:

        url = '%sget.php?version=%s&%s=%s' % (
            base_url,
            __version__,
            args.command,
            args.name,
        )
        print("Contacting server (%s)..." % base_url)
        response = urlopen(url)
        match, score = response.read().decode().split(',')
        response.close()

        if int(score) == 0 or ui.yes_or_no('Do you mean \'%s\'?' % match):
            download(args.command, match + '.npz')

    else:

        print('Contacting server (%s)...' % base_url)
        response = urlopen(
            '%sget.php?version=%s&%s' % (base_url, __version__, args.command)
        )
        line = response.readlines()
        response.close()

        print('')
        print('Available %ss:' % args.command)

        print('{:<2} {:<25}    {:>4}'.format('ID', 'Name', 'Size'))
        print('-' * 36)

        items = line[0].split(b';')
        for i, item in enumerate(items):
            name, size = item.split(b',')
            size = int(size) / 1024 ** 2  # Bytes to MBytes

            print('{:>2d} {:<25} {:>4d} MB'.format(i, name.decode("utf-8"), int(size)))
        print('')

        down_list = raw_input(
            'Please list which datasets to download (e.g. 0 1 2 6) or type \'all\': '
        )
        down_idxs = []
        if 'all' in down_list.lower():
            down_idxs = list(range(len(items)))
        elif re.match(
            "^ *[0-9][0-9 ]*$", down_list
        ):  # only digits and spaces, at least one digit
            down_idxs = [int(idx) for idx in re.split(r'\s+', down_list.strip())]
            down_idxs = list(set(down_idxs))
        else:
            print(' ABORTED.')

        for idx in down_idxs:
            if idx not in range(len(items)):
                print(
                    ui.warn_str('[WARN]')
                    + ' Index '
                    + str(idx)
                    + ' out of range, skipping.'
                )
            else:
                name = items[idx].split(b',')[0].decode("utf-8")
                if os.path.exists(name):
                    print("'%s' exists, skipping." % (name))
                    continue

                download(args.command, name + '.npz')
    print('')


if __name__ == "__main__":
    main()
