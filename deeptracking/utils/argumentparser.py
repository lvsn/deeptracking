"""
    Standard argument parser for various scripts

    date : 2017-03-21
"""

__author__ = "Mathieu Garon"
__version__ = "0.0.1"

import getopt


class ArgumentParser:
    def __init__(self, arguments):
        self.help = False
        self.verbose = False
        self.config_file = None
        try:
            opts, args = getopt.getopt(arguments, "hc:v", ["config="])
        except getopt.GetoptError:
            self.help = True
        for opt, arg in opts:
            if opt in ('-c', "--config"):
                self.config_file = arg
            if opt in ('-v'):
                self.verbose = True
            if opt in ('-h'):
                self.help = True
        # check if c option was there, if not trigger help message
        if self.config_file is None:
            self.help = True

    def print_help(self):
        print("Help :\n"
              "\t-c : config file\n"
              "\t-h : this help\n"
              "\t-v : verbose")
