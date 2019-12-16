#!/usr/bin/python

# MIT License
#
# Copyright (c) 2018-2019 Stefan Chmiela
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


def progr_bar(current, total, disp_str='', sec_disp_str=None):
    """
    Print progress bar.

    Example:
    ``[ 45%] Task description (secondary string)``

    Parameters
    ----------
        current : int
            How many items already processed?
        total : int
            Total number of items?
        disp_str : :obj:`str`, optional
            Task description.
        sec_disp_str : :obj:`str`, optional
            Additional string shown in gray.
    """

    is_done = np.isclose(current - total, 0.0)
    progr = float(current) / total

    str_color = pass_str if is_done else info_str
    sys.stdout.write(('\r' + str_color('[%3d%%]') + ' %s') % (progr * 100, disp_str))

    if sec_disp_str is not None:
        w = MAX_PRINT_WIDTH - LOG_LEVELNAME_WIDTH - len(disp_str) - 1
        sys.stdout.write(' \x1b[90m{0: >{width}}\x1b[0m'.format(sec_disp_str, width=w))

    if is_done:
        sys.stdout.write('\n')

    sys.stdout.flush()


def progr_toggle(is_done, disp_str='', sec_disp_str=None):
    """
    Print progress toggle.

    Example (not done):
    ``[ .. ] Task description (secondary string)``

    Example (done):
    ``[DONE] Task description (secondary string)``

    Parameters
    ----------
        is_done : bool
            Task done?
        disp_str : :obj:`str`, optional
            Task description.
        sec_disp_str : :obj:`str`, optional
            Additional string shown in gray.
    """

    sys.stdout.write(
        '\r%s '
        % (
            pass_str('[DONE]')
            if is_done
            else info_str('[') + info_str(blink_str(' .. ')) + info_str(']')
        )
    )
    sys.stdout.write(disp_str)

    if sec_disp_str is not None:
        w = MAX_PRINT_WIDTH - LOG_LEVELNAME_WIDTH - len(disp_str) - 1
        sys.stdout.write(' \x1b[90m{0: >{width}}\x1b[0m'.format(sec_disp_str, width=w))

    if is_done:
        sys.stdout.write('\n')

    sys.stdout.flush()


# COLORS

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
COLOR_SEQ, RESET_SEQ = '\033[{:d};{:d};{:d}m', '\033[0m'


def color_str(str, fore_color=WHITE, back_color=BLACK, bold=False):

    # foreground is set with 30 plus the number of the color, background with 40
    return (
        COLOR_SEQ.format(1 if bold else 0, 30 + fore_color, 40 + back_color)
        + str
        + RESET_SEQ
    )


def white_back_str(str):
    return color_str(str, fore_color=BLACK, back_color=WHITE, bold=True)
    # return '\x1b[1;7m' + str + '\x1b[0m'


# def green_back_str(str):
# return color_str(str, back_color=GREEN, bold=True)
#    return '\x1b[1;30;42m' + str + '\x1b[0m'


def yellow_back_str(str):
    return '\x1b[1;30;43m' + str + '\x1b[0m'


def white_bold_str(str):
    return '\x1b[1;37m' + str + '\x1b[0m'


def gray_str(str):
    return '\x1b[90m' + str + '\x1b[0m'


def underline_str(str):
    return '\x1b[4m' + str + '\x1b[0m'


def blink_str(str):
    return '\x1b[5m' + str + '\x1b[0m'


def info_str(str):
    return '\x1b[1;37m' + str + '\x1b[0m'


def pass_str(str):
    return color_str(str, fore_color=GREEN, bold=True)
    # return '\x1b[1;32m' + str + '\x1b[0m'


# def warn_str(str):
#    return '\x1b[1;33m' + str + '\x1b[0m'


# def fail_str(str):
#    return '\x1b[1;31m' + str + '\x1b[0m'


# def warning(str):
#    print(ui.WARN_str('[WARN]') + ' %s' % err)

# def is_lattice_supported(lat):  # TODO: remove me

#     is_supported = False
#     if (
#         np.all(lat == np.diag(np.diagonal(lat)))
#         and len(set(np.diag(lat)))  # diagonal matrix?
#         == 1  # all diagonal elements all the same?
#     ):
#         is_supported = True

#     return is_supported


def unicode_str(s):

    if sys.version[0] == '3':
        return str(s, 'utf-8', 'ignore')
    else:
        return str(s)


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
        x
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

    # arr_min, arr_max = np.min(arr), np.max(arr)
    return '{:<.3} |-- {:^8.3} --| {:<9.3}'.format(min, max - min, max)


def print_step_title(title_str, sec_title_str='', underscore=True):

    if sec_title_str != '':
        sec_title_str = ' ' + sec_title_str

    underscore_str = '-' * MAX_PRINT_WIDTH if underscore else ''

    print(
        '\n'
        + white_back_str(' ' + title_str + ' ')
        + sec_title_str
        + '\n'
        + underscore_str
    )


def print_two_column_str(str, sec_str=''):

    print(
        '{} \x1b[90m{:>{width}}\x1b[0m'.format(
            str, sec_str, width=MAX_PRINT_WIDTH - len(str) - 1
        )
    )


def print_lattice(lat=None):

    from . import io

    lat_str = 'n/a'
    if lat is not None:
        lat_str = gen_lattice_str(lat)
        lengths, angles = io.lattice_vec_to_par(lat)

    print('  {:<18} {}'.format('Lattice:', lat_str))
    if lat is not None:
        print('    {:<16} a = {:g}, b = {:g}, c = {:g}'.format('Lengths:', *lengths))
        print(
            '    {:<16} alpha = {:g}, beta = {:g}, gamma = {:g}'.format(
                'Angles [deg]:', *angles
            )
        )
