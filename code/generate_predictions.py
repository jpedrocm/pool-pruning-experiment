###############################################################################
import numpy as np
import random as rn

#DO NOT CHANGE THIS
np.random.seed(1478)
rn.seed(2264)
###################

from utils import load_datasets_filenames, load_experiment_configuration
from utils import load_dataset, save_predictions
from utils import select_validation_set

from sklearn.model_selection import StratifiedKFold



if __name__ == "__main__":

	print "Loading configurations"

	datasets_filenames = load_datasets_filenames()
	config = load_experiment_configuration()
	predictions = {}
	exp = 1

	print "Starting experiment"

	for dataset_filename in datasets_filenames:
		instances, gold_labels = load_dataset(dataset_filename)
		skfold = StratifiedKFold(n_splits = config["num_folds"],
			                     shuffle = True)

		gold_labels = (gold_labels["defects"] == 'true').astype(int)

		predictions[dataset_filename] = {}

		for fold, division in enumerate(skfold.split(X=instances, y=gold_labels), 1):
			train_idxs = division[0]
			test_idxs = division[1]
			train_instances = instances.iloc[train_idxs].values
			train_gold_labels = gold_labels.iloc[train_idxs].values.ravel()
			test_instances = instances.iloc[test_idxs].values
			test_gold_labels = gold_labels.iloc[test_idxs].values.ravel()

			predictions[dataset_filename][fold] = {}
			predictions[dataset_filename][fold]["gold_labels"] = test_gold_labels.tolist()

			for hardness_type, filter_func in config["validation_hardnesses"]:

				validation_instances, validation_gold_labels = select_validation_set(
					  train_instances, train_gold_labels, filter_func, config["kdn"])

				predictions[dataset_filename][fold][hardness_type] = {}
				subpredictions = predictions[dataset_filename][fold][hardness_type]

				base_clf = config["base_classifier"]()
				clf_pool = config["generation_strategy"](base_clf, config["pool_size"])
				clf_pool.fit(train_instances, train_gold_labels)

				for strategy_name, pruning_strategy in config["pruning_strategies"]:

					pruned_pool, pool_rest = pruning_strategy(clf_pool,
						                          validation_instances,
						                          validation_gold_labels)

					cur_predictions = pruned_pool.predict(test_instances)
					subpredictions[strategy_name] = (cur_predictions.astype(int).tolist(),
						                             pool_rest)

					print "Experiment " + str(exp)
					exp+=1

	print "Finished experiment"
	save_predictions(predictions)
	print "Stored predictions"