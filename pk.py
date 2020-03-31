#!/usr/bin/python

"""
Convert pickle data to human-readable form.

Usage:
  pk.py inputFile [outputFile]
"""
import pandas as pd
import sys
import pickle as pkl

pd.set_option('display.max_rows', 1000)
    
def main(args):
    try:
        inputFile = open(args[1], 'rb')
        input = pkl.load(inputFile)
        inputFile.close()
    except IndexError:
        usage()
        return False

    print(input)

    return True


def usage():
    print(__doc__)


if __name__ == "__main__":
    sys.exit(not main(sys.argv))
