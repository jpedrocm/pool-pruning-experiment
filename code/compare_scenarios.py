###############################################################################
import argparse

from utils import read_pandas_summary, separate_pandas_summary
from utils import write_comparison, bool_str



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--separate", type = bool_str, default="True", 
		                help="Analyze datasets separately. Default: True.")
	parser.add_argument("-c1", "--column1", type = str, default = "strategy", 
		                help="First column to focus. Default: strategy")
	parser.add_argument("-c2", "--column2", type = str, default = None, 
		                help="Second column to focus. Default: None.")
	parser.add_argument("-f", "--filename", type=str, default = "scenarios",
		                help="Output filename without extension")
	args = parser.parse_args()

	print "Step 1 - Reading summary"
	df = read_pandas_summary()

	print "Step 2 - Comparing scenarios"
	dfs = separate_pandas_summary(df, args.separate)

	print "Step 3 - Storing comparisons"
	focus_columns = [col for col in [args.column1, args.column2] if col]
	write_comparison(dfs, focus_columns, args.filename)