'''
Author:	 Iskandar Pashayev
Purpose: Provide a wrapper for an option parser to parse command line arguments
'''

from optparse import OptionParser

class Parser(object):
    def __init__(self):
        parser = OptionParser()
        '''
        Removing this option - set in configuration file now.
        parser.add_option('-f','--num-false',
                          action='store',
                          type='int',
                          dest='num_false',
                          default='1',
                          help='generate NUM_FALSE false examples per pdf scan',
                          metavar='NUM_FALSE')
        '''
        parser.add_option('-c','--config-file',
                          dest='config_file',
                          help='configure neural network from CONFIG_FILE',
                          metavar='CONFIG_FILE')
        parser.add_option('-o', '--output-file',
                          dest='output_file',
                          help='write report to OUTPUT_FILE',
                          metavar='OUTPUT_FILE')
        parser.add_option('-q','--quiet',
                          action='store_true',
                          dest='verbose',
                          default=False,
                          help='suppress output to stdout')
    def get_args(self):
        (self.options, self.args) = parser.parse_args()
    def read(self):
        self.config_dict = { 'hidden_layers':[] }
        with open self.options.config_file as cfg:
            for line in cfg:
                key, value = line.rstrip().split(' ')
                if key == 'hidden_layer':
                    self.config_dict[key].append(value)
                else:
                    self.config_dict[key] = value

    # Getter methods for classification
    def get_config_file_path(self): return self.options.config_file
    def get_output_file_path(self): return self.options.output_file
    def get_verbose(self): return self.options.verbose
    def get_data_path(self): return self.config_dict['data_path']
    def get_num_iter(self): return self.config_dict['num_iterations']
    def get_num_false(self): return self.config_dict['num_false_per_pdf']
    def get_hidden(self): return self.config_dict['hidden_layers']
    

if __name__ == '__main__':
    parser = Parser(); parser.get_args()
    print 'The configuration file is:', parser.get_config_file_path()
    print 'The output file is:', parser.get_output_file_path()
    print 'Suppressing console output:', parser.get_verbose()

