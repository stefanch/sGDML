#!/usr/bin/python

# MIT License
#
# Copyright (c) 2018-2021 Stefan Chmiela
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
from functools import partial

from .. import __version__, MAX_PRINT_WIDTH, LOG_LEVELNAME_WIDTH
import textwrap
import re
import sys

if sys.version[0] == '3':
    raw_input = input

import numpy as np


def yes_or_no(question):
    """
    Ask for yes/no user input on a question.

    Any response besides ``y`` yields a negative answer.

    Parameters
    ----------
        question : :obj:`str`
            User question.
    """

    reply = raw_input(question + ' (y/n): ').lower().strip()
    if not reply or reply[0] != 'y':
        return False
    else:
        return True


last_callback_pct = 0


def callback(
    current,
    total=1,
    disp_str='',
    sec_disp_str=None,
    done_with_warning=False,
    newline_when_done=True,
):
    """
    Print progress or toggle bar.

    Example (progress):
    ``[ 45%] Task description (secondary string)``

    Example (toggle, not done):
    ``[ .. ] Task description (secondary string)``

    Example (toggle, done):
    ``[DONE] Task description (secondary string)``

    Parameters
    ----------
        current : int
            How many items already processed?
        total : int, optional
            Total number of items? If there is only
            one item, the toggle style is used.
        disp_str : :obj:`str`, optional
            Task description.
        sec_disp_str : :obj:`str`, optional
            Additional string shown in gray.
        done_with_warning : bool, optional
            Indicate that the process did not
            finish successfully.
        newline_when_done : bool, optional
            Finish with a newline character once
            current=total (default: True)?
    """

    global last_callback_pct

    is_toggle = total == 1
    is_done = np.isclose(current - total, 0.0)

    bold_color_str = partial(color_str, bold=True)

    if is_toggle:

        if is_done:
            if done_with_warning:
                flag_str = bold_color_str('[WARN]', fore_color=YELLOW)
            else:
                flag_str = bold_color_str('[DONE]', fore_color=GREEN)

        else:
            flag_str = bold_color_str('[' + blink_str(' .. ') + ']')
    else:

        # Only show progress in 10 percent steps when not printing to terminal.
        pct = int(float(current) * 100 / total)
        pct = int(np.ceil(pct / 10.0)) * 10 if not sys.stdout.isatty() else pct

        # Do not print, if there is no need to.
        if not is_done and pct == last_callback_pct:
            return
        else:
            last_callback_pct = pct

        flag_str = bold_color_str(
            '[{:3d}%]'.format(pct), fore_color=GREEN if is_done else WHITE
        )

    sys.stdout.write('\r{} {}'.format(flag_str, disp_str))

    if sec_disp_str is not None:
        w = MAX_PRINT_WIDTH - LOG_LEVELNAME_WIDTH - len(disp_str) - 1
        # sys.stdout.write(' \x1b[90m{0: >{width}}\x1b[0m'.format(sec_disp_str, width=w))
        sys.stdout.write(
            color_str(' {:>{width}}'.format(sec_disp_str, width=w), fore_color=GRAY)
        )

    if is_done and newline_when_done:
        sys.stdout.write('\n')

    sys.stdout.flush()


# use this to integrate a callback for a subtask with an existing callback function
# 'subtask_callback = partial(ui.sec_callback, main_callback=self.callback)'
def sec_callback(
    current, total=1, disp_str=None, sec_disp_str=None, main_callback=None, **kwargs
):
    global last_callback_pct

    assert main_callback is not None

    is_toggle = total == 1
    is_done = np.isclose(current - total, 0.0)

    sec_disp_str = disp_str
    if is_toggle:
        sec_disp_str = '{} | {}'.format(disp_str, 'DONE' if is_done else ' .. ')
    else:

        # Only show progress in 10 percent steps when not printing to terminal.
        pct = int(float(current) * 100 / total)
        pct = int(np.ceil(pct / 10.0)) * 10 if not sys.stdout.isatty() else pct

        # Do not print, if there is no need to.
        if pct == last_callback_pct:
            return

        last_callback_pct = pct
        sec_disp_str = '{} | {:3d}%'.format(disp_str, pct)

    main_callback(0, sec_disp_str=sec_disp_str, **kwargs)


# COLORS

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, GRAY = list(range(8)) + [60]
COLOR_SEQ, RESET_SEQ = '\033[{:d};{:d};{:d}m', '\033[0m'

ENABLE_COLORED_OUTPUT = (
    sys.stdout.isatty()
)  # Running in a real terminal or piped/redirected?


def color_str(str, fore_color=WHITE, back_color=BLACK, bold=False):

    if ENABLE_COLORED_OUTPUT:

        # foreground is set with 30 plus the number of the color, background with 40
        return (
            COLOR_SEQ.format(1 if bold else 0, 30 + fore_color, 40 + back_color)
            + str
            + RESET_SEQ
        )
    else:
        return str


def blink_str(str):

    return '\x1b[5m' + str + '\x1b[0m' if ENABLE_COLORED_OUTPUT else str


def unicode_str(s):

    if sys.version[0] == '3':
        s = str(s, 'utf-8', 'ignore')
    else:
        s = str(s)

    return s.rstrip('\x00')  # remove null-characters


def gen_memory_str(bytes):

    pwr = 1024
    n = 0
    pwr_strs = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while bytes > pwr and n < 4:
        bytes /= pwr
        n += 1

    return '{:.{num_dec_pts}f} {}B'.format(
        bytes, pwr_strs[n], num_dec_pts=max(0, n - 2)
    )  # 1 decimal point for GB, 2 for TB


def gen_lattice_str(lat):

    lat_str, col_widths = gen_mat_str(lat)
    desc_str = (' '.join([('{:' + str(w) + '}') for w in col_widths])).format(
        'a', 'b', 'c'
    ) + '\n'

    lat_str = indent_str(lat_str, 21)

    return desc_str + lat_str


def str_plen(str):
    """
    Returns printable length of string. This function can only account for invisible characters due to string styling with ``color_str``.

    Parameters
    ----------
        str : :obj:`str`
            String.

    Returns
    -------
        :obj:`str`

    """

    num_colored_subs = str.count(RESET_SEQ)
    return len(str) - (
        14 * num_colored_subs
    )  # 14: length of invisible characters per colored segment


def wrap_str(str, width=MAX_PRINT_WIDTH - LOG_LEVELNAME_WIDTH):
    """
    Wrap multiline string after a given number of characters. The default maximum line already accounts for the indentation due to the logging level label.

    Parameters
    ----------
        str : :obj:`str`
            Multiline string.
        width : int, optional
            Max number of characters in a line.

    Returns
    -------
        :obj:`str`

    """

    return '\n'.join(
        [
            '\n'.join(
                textwrap.wrap(
                    line,
                    width + (len(line) - str_plen(line)),
                    break_long_words=False,
                    replace_whitespace=False,
                )
            )
            for line in str.splitlines()
        ]
    )


def indent_str(str, indent):
    """
    Indents all lines of a multiline string right by a given number of
    characters.

    Parameters
    ----------
        str : :obj:`str`
            Multiline string.
        indent : int
            Number of characters added in front of each line.

    Returns
    -------
        :obj:`str`

    """

    return re.sub('^', ' ' * indent, str, flags=re.MULTILINE)


def wrap_indent_str(label, str, width=MAX_PRINT_WIDTH - LOG_LEVELNAME_WIDTH):
    """
    Wraps and indents a multiline string to arrange it with the provided label in two columns. The default maximum line already accounts for the indentation due to the logging level label.

    Example:
    ``<label><multiline string>``

    Parameters
    ----------
        label : :obj:`str`
            Label
        str : :obj:`str`
            Multiline string.

    Returns
    -------
        :obj:`str`

    """

    label_len = str_plen(label)

    str = wrap_str(str, width - label_len)
    str = indent_str(str, label_len)

    return label + str[label_len:]


def merge_col_str(
    col_str1, col_str2
):  # merge two multiline strings that represent columns in a table
    """
    Merges two multiline strings that represent columns in a table by
    concatenating each pair of lines.

    Note
    ----
        Both strings must have the same number of lines.

    Parameters
    ----------
        col_str1 : :obj:`str`
            First multiline string.
        col_str2 : :obj:`str`
            Second multiline string.

    Returns
    -------
        :obj:`str`

    """

    return '\n'.join(
        [
            ' '.join([c1, c2])
            for c1, c2 in zip(col_str1.split('\n'), col_str2.split('\n'))
        ]
    )


def gen_mat_str(mat):
    """
    Converts a matrix to a multiline string such that the decimal points
    align in each column. Trailing zeros are replaced with spaces.

    Parameters
    ----------
        mat : :obj:`numpy.ndarray`

    Returns
    -------
        :obj:`str`
            String representation of matrix.

    """

    def _int_len(
        x,
    ):  # length of string representation before decimal point (including sign)
        return len(str(int(abs(x)))) + (0 if x >= 0 else 1)

    def _dec_len(x):  # length of string representation after decimal point

        x_str_split = '{:g}'.format(x).split('.')
        return len(x_str_split[1]) if len(x_str_split) > 1 else 0

    def _max_int_len_for_col(
        mat, col
    ):  # length of string representation before decimal point for each col
        col_min = np.min(mat[:, col])
        col_max = np.max(mat[:, col])
        return max(_int_len(col_min), _int_len(col_max))

    def _max_dec_len_for_col(
        mat, col
    ):  # length of string representation after decimal point for each col
        return max([_dec_len(cell) for cell in mat[:, col]])

    n_cols = mat.shape[1]
    col_int_widths = [_max_int_len_for_col(mat, i) for i in range(n_cols)]
    col_dec_widths = [_max_dec_len_for_col(mat, i) for i in range(n_cols)]
    col_widths = [iw + cd + 1 for iw, cd in zip(col_int_widths, col_dec_widths)]

    mat_str = ''
    for row in mat:
        if mat_str != '':
            mat_str += '\n'
        mat_str += ' '.join(
            ' ' * max(col_int_widths[j] - _int_len(x), 0)
            + ('{: <' + str(_int_len(x) + col_dec_widths[j] + 1) + 'g}').format(x)
            for j, x in enumerate(row)
        )

    return mat_str, col_widths


def gen_range_str(min, max):
    """
    Generates a string that shows a minimum and maximum value, as well as the range.

    Example:
    ``<min> |-- <range> --| <max>``

    Parameters
    ----------
        min : float
            Minimum value.
        max : float
            Maximum value.

    Returns
    -------
        :obj:`str`

    """

    return '{:<.3f} |-- {:^8.3f} --| {:<9.3f}'.format(min, max - min, max)


def print_step_title(title_str, sec_title_str='', underscore=True):

    if sec_title_str != '':
        sec_title_str = ' ' + sec_title_str

    underscore_str = '\n' + '-' * MAX_PRINT_WIDTH if underscore else ''

    print(
        '\n'
        + color_str(
            ' ' + title_str + ' ', fore_color=BLACK, back_color=WHITE, bold=True
        )
        + sec_title_str
        + underscore_str
    )


def print_two_column_str(str, sec_str=''):

    sec_str = color_str(
        '{:>{width}}'.format(sec_str, width=MAX_PRINT_WIDTH - str_plen(str) - 1),
        fore_color=GRAY,
    )
    print('{} {}'.format(str, sec_str))

    # print(
    #     '{} \x1b[90m{:>{width}}\x1b[0m'.format(
    #         str, sec_str, width=MAX_PRINT_WIDTH - str_plen(str) - 1
    #     )
    # )


def print_lattice(lat=None, inset=False):

    from . import io

    lat_str = 'n/a'
    if lat is not None:
        lat_str = gen_lattice_str(lat)
        lengths, angles = io.lattice_vec_to_par(lat)

    if inset:
        print('    {:<16} {}'.format('Lattice:', lat_str))
    else:
        print('  {:<18} {}'.format('Lattice:', lat_str))
    if lat is not None:
        print('    {:<16} a = {:g}, b = {:g}, c = {:g}'.format('Lengths:', *lengths))
        print(
            '    {:<16} alpha = {:g}, beta = {:g}, gamma = {:g}'.format(
                'Angles [deg]:', *angles
            )
        )
