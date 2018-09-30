###############################################################################
from utils import load_predictions, generate_metrics, save_pandas_summary
from utils import summarize_metrics_folds, pandanize_summary


if __name__ == "__main__":

	print "Loading data"
	predictions_dict = load_predictions()

	print "Started calculation"
	metrics_dict = generate_metrics(predictions_dict)
	summary = summarize_metrics_folds(metrics_dict)
	print "Finished calculation"

	pd_summary = pandanize_summary(summary)
	save_pandas_summary(pd_summary)
	print "Stored metrics"