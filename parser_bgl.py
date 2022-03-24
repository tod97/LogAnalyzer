""" 
    Use this parser to produce the log_structured starting from a raw BGL log file
"""

from logparser import Drain, AEL
dataset_path = './'
input_dir  = dataset_path + 'loghub/BGL/'  # The input directory of log file
output_dir = dataset_path = 'loghub/BGL/'  # The output directory of parsing results
log_file   = 'BGL_2k.log'  # The input log file name
log_format = '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>'  # HDFS log format

""" # Regular expression list for optional preprocessing (default: [])
regex      = [
    r'blk_(|-)[0-9]+' , # block id
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
]
st         = 0.5  # Similarity threshold
depth      = 4  # Depth of all leaf nodes

parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
parser.parse(log_file) """

minEventCount = 2 # The minimum number of events in a bin
merge_percent = 0.5 # The percentage of different tokens 
regex         = [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'] # Regular expression list for optional preprocessing (default: [])

parser = AEL.LogParser(input_dir, output_dir, log_format, rex=regex, minEventCount=minEventCount, merge_percent=merge_percent)
parser.parse(log_file) 