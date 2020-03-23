#!/usr/bin/python

# MIT License
#
# Copyright (c) 2019 Stefan Chmiela
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

__version__ = '0.4.2.dev0'

MAX_PRINT_WIDTH = 100
LOG_LEVELNAME_WIDTH = 7  # do not modify


# Logging

import copy
import logging
import re
import textwrap

from .utils import ui


class ColoredFormatter(logging.Formatter):

    LEVEL_COLORS = {
        'DEBUG': (ui.CYAN, ui.BLACK),
        'INFO': (ui.WHITE, ui.BLACK),
        'DONE': (ui.GREEN, ui.BLACK),
        'WARNING': (ui.YELLOW, ui.BLACK),
        'ERROR': (ui.RED, ui.BLACK),
        'CRITICAL': (ui.BLACK, ui.RED),
    }

    LEVEL_NAMES = {
        'DEBUG': '[DEBG]',
        'INFO': '[INFO]',
        'DONE': '[DONE]',
        'WARNING': '[WARN]',
        'ERROR': '[FAIL]',
        'CRITICAL': '[CRIT]',
    }

    def __init__(self, msg, use_color=True):

        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):

        _record = copy.copy(record)
        levelname = _record.levelname
        msg = _record.msg

        levelname = ui.color_str(
            self.LEVEL_NAMES[levelname],
            self.LEVEL_COLORS[levelname][0],
            self.LEVEL_COLORS[levelname][1],
            bold=True,
        )

        # wrap long messages
        msg = ui.wrap_str(msg)

        # indent multiline strings after the first line
        msg = ui.indent_str(msg, LOG_LEVELNAME_WIDTH)[LOG_LEVELNAME_WIDTH:]

        _record.levelname = levelname
        _record.msg = msg
        return logging.Formatter.format(self, _record)


class ColoredLogger(logging.Logger):
    def __init__(self, name):

        logging.Logger.__init__(self, name, logging.DEBUG)

        # add 'DONE' logging level
        logging.DONE = logging.INFO + 1
        logging.addLevelName(logging.DONE, 'DONE')

        # only display levelname and message
        formatter = ColoredFormatter('%(levelname)s %(message)s')

        # this handler will write to sys.stderr by default
        hd = logging.StreamHandler()
        hd.setFormatter(formatter)
        hd.setLevel(
            logging.DEBUG
        )  # control logging level here (default: logging.DEBUG)

        self.addHandler(hd)
        return

    def done(self, msg, *args, **kwargs):

        if self.isEnabledFor(logging.DONE):
            self._log(logging.DONE, msg, args, **kwargs)


logging.setLoggerClass(ColoredLogger)
