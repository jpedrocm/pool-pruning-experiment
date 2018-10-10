###############################################################################
from utils import load_predictions_data, generate_metrics, save_pandas_summary
from utils import summarize_metrics_folds, pandanize_summary


if __name__ == "__main__":

	print "Step 1 - Loading data"
	predictions_dict = load_predictions_data()

	print "Step 2 - Started calculation"
	metrics_dict = generate_metrics(predictions_dict)
	summary = summarize_metrics_folds(metrics_dict)
	print "Step 2 - Finished calculation"

	print "Step 3 - Storing metrics"
	pd_summary = pandanize_summary(summary)
	save_pandas_summary(pd_summary)