'''
Author:	 Iskandar Pashayev
Purpose: Provide a wrapper for an option parser to parse command line arguments
'''

from optparse import OptionParser

class Parser(object):
    def __init__(self):
        parser = OptionParser()
        parser.add_option('-f','--num-false',
                          action='store',
                          type='int',
                          dest='num_false',
                          default='1',
                          help='generate NUM_FALSE false examples per pdf scan',
                          metavar='NUM_FALSE')
        parser.add_option('-c','--config-file',
                          dest='config_file',
                          help='configure neural network from CONFIG_FILE',
                          metavar='CONFIG_FILE')
        parser.add_option('-o', '--output-file',
                          dest='output_file',
                          help='write report to OUT_FILE',
                          metavar='OUT_FILE')
        parser.add_option('-q','--quiet',
                          action='store_false',
                          dest='verbose',
                          default=True,
                          help='suppress output to stdout')
        (self.options, self.args) = parser.parse_args()
    def get_args(self):
        return (self.options.num_false,
                self.options.config_file,
                self.options.output_file,
                self.options.verbose)


if __name__ == '__main__':
    parser = Parser()
    num_false, config_file, output_file, verbose = parser.get_args()
    print 'Generating', num_false, 'examples per pdf scan'
    print 'The configuration file is:', config_file
    print 'The output file is:', output_file
    print 'Suppressing console output:', verbose
