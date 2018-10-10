###############################################################################
from functools import partial
from math import sqrt
from copy import deepcopy
import operator, sys

import json
import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

from prefit_voting_classifier import VotingClassifier


def load_experiment_configuration():
	STRATEGY_PERCENTAGE = 0.5
	N_JOBS = -1
	PRUNNING_CLUSTERS = 10

	config = {
	"num_folds": 10,
	"pool_size": 100,
	"kdn": 5,
	"strategy_percentage": STRATEGY_PERCENTAGE,
	"validation_hardnesses": _create_validation_hardnesses(threshold = 0.5),
	"base_classifier": partial(Perceptron, max_iter = 10, tol = 0.001,
		                       penalty = None, n_jobs = N_JOBS),
	"generation_strategy": partial(BaggingClassifier, 
		                           max_samples = STRATEGY_PERCENTAGE,
		                           n_jobs = -1),
	"pruning_strategies": _create_pruning_strategies(PRUNNING_CLUSTERS,
		                                             N_JOBS),
	"diversity_measures": _create_diversity_measures()
	}

	return config

def _create_validation_hardnesses(threshold):
	return [("None", partial(operator.gt, 2)), 
	        ("Hard", partial(operator.lt, threshold)), 
	        ("Easy", partial(operator.gt, threshold))]

def _create_diversity_measures():
	return []

def _create_pruning_strategies(num_clusters, n_jobs):
	return [("Best First", partial(_best_first_pruning)),
	        ("K Best Means", partial(_k_best_means, k=num_clusters, 
	        	                     n_jobs = n_jobs))]

def _find_k_neighbours(distances, k):
	
	matrix_neighbours = []
	for i in xrange(len(distances)):
		
		cur_neighbours = set()
		while len(cur_neighbours) < k:
			min_ix = np.argmin(distances[i])
			distances[i, min_ix] = sys.float_info.max

			if min_ix != i:
				cur_neighbours.add(min_ix)

		matrix_neighbours.append(list(cur_neighbours))

	return matrix_neighbours

def _calculate_kdn_hardness(instances, gold_labels, k):
	distances = euclidean_distances(instances, instances)
	neighbours = _find_k_neighbours(distances, k)

	hards = []
	for i in xrange(len(neighbours)):
		fixed_label = gold_labels[i]
		k_labels = gold_labels[neighbours[i]]
		dn = sum(map(lambda label: label != fixed_label, k_labels))
		hards.append(float(dn)/k)

	return hards

def select_validation_set(instances, labels, operator, k):
	hards = _calculate_kdn_hardness(instances, labels, k)
	filtered_triples = _filter_based_hardness(instances, labels, hards, operator)
	validation_instances = [t[0] for t in filtered_triples]
	validation_labels = [t[1] for t in filtered_triples]
	return np.array(validation_instances), validation_labels

def _filter_based_hardness(instances, labels, hards, op):
	triples = [(instances[i], labels[i], hards[i]) for i in xrange(len(hards))]
	return filter(lambda t: op(t[2]), triples)

def _order_clfs(pool_clf, validation_instances, validation_labels):
	clfs = pool_clf.estimators_
	clfs_feats = pool_clf.estimators_features_
	predictions = [clf.predict(validation_instances) for clf in clfs]
	errors = [_error_score(validation_labels, predicted_labels) for predicted_labels in predictions]
	triples = [(clfs[i], clfs_feats[i], errors[i]) for i in xrange(len(errors))]
	return sorted(triples, key=lambda t: t[2])

def _find_k_clusters(pool_clf, k, n_jobs):
	clfs = pool_clf.estimators_
	clfs_feats = pool_clf.estimators_features_

	pool_weights = [clf.coef_[0] for clf in clfs]
	k_means = KMeans(n_clusters = k, n_jobs = n_jobs)
	clusters_labels = k_means.fit_predict(pool_weights)

	clusters = {cluster_label: [] for cluster_label in clusters_labels}
	for i in xrange(len(clfs)):
		cluster = clusters_labels[i]
		clusters[cluster].append((clfs[i], clfs_feats[i]))

	return clusters

def _find_best_per_cluster(clusters, validation_instances, validation_labels):
	best_k_clf = []
	best_k_feats = []

	for cluster, clfs_tuples in clusters.iteritems():
		cur_best_clf = None
		cur_best_feats = None
		cur_best_error = 100

		for clf_tuple in clfs_tuples:
			clf = clf_tuple[0]
			predicted = clf.predict(validation_instances)
			error = _error_score(validation_labels, predicted)

			if error < cur_best_error:
				cur_best_error = error
				cur_best_clf = clf
				cur_best_feats = clf_tuple[1]

		best_k_clf.append(cur_best_clf)
		best_k_feats.append(cur_best_feats)

	return _get_voting_clf(best_k_clf, best_k_feats), len(best_k_clf)

def _k_best_means(pool_clf, validation_instances, validation_labels, k, n_jobs):
	clusters = _find_k_clusters(pool_clf, k, n_jobs)
	return _find_best_per_cluster(clusters, validation_instances, validation_labels)

def _find_best_first(triples, validation_instances, validation_labels):
	best_ensemble_error = 100
	best_ensemble = None
	best_ensemble_size = 0

	cur_clfs = []
	cur_feats = []
	for triple in triples:
		clf, clf_feat, error = triple
		cur_clfs.append(clf)
		cur_feats.append(clf_feat)
		ensemble = _get_voting_clf(cur_clfs, cur_feats)
		predicted = ensemble.predict(validation_instances)
		error = _error_score(validation_labels, predicted)

		if error < best_ensemble_error:
			best_ensemble_error = error
			best_ensemble = ensemble
			best_ensemble_size = len(cur_clfs)

	return best_ensemble, best_ensemble_size

def _best_first_pruning(pool_clf, validation_instances, validation_labels):
	ordered_triples = _order_clfs(pool_clf, validation_instances, 
		                          validation_labels)

	return _find_best_first(ordered_triples, validation_instances, 
		                    validation_labels)

def _get_voting_clf(base_clfs, clfs_feats):
	pool_size = len(base_clfs)
	clfs_tuples = [(str(i), base_clfs[i]) for i in xrange(pool_size)]
	return VotingClassifier(clfs_tuples, clfs_feats, voting = 'hard')

def load_datasets_filenames():
	filenames = ["cm1", "jm1"]
	return filenames

def load_dataset(set_filename):
	SET_PATH = "../data/"
	FILETYPE = ".arff"
	full_filepath = SET_PATH + set_filename + FILETYPE

	data, _ = arff.loadarff(full_filepath)

	dataframe = pd.DataFrame(data)
	dataframe.dropna(inplace=True)

	gold_labels = pd.DataFrame(dataframe["defects"])
	instances = dataframe.drop(columns = "defects")

	return instances, gold_labels

def save_predictions(data):
	with open('../predictions/all_predictions.json', 'w') as outfile:
		json.dump(data, outfile)

def load_predictions_data():
	with open('../predictions/all_predictions.json', 'r') as outfile:
		return json.load(outfile)

def _error_score(gold_labels, predicted_labels):
	return 1 - accuracy_score(gold_labels, predicted_labels)

def _g1_score(gold_labels, predicted_labels, average):
	precision = precision_score(gold_labels, predicted_labels, average=average)
	recall = recall_score(gold_labels, predicted_labels, average=average)
	return sqrt(precision*recall)

def _calculate_metrics(gold_labels, data):

	predicted_labels = data[0]
	final_pool_size = data[1]

	metrics = {}
	metrics["auc_roc"] = roc_auc_score(gold_labels, predicted_labels, average='macro')
	metrics["g1"] = _g1_score(gold_labels, predicted_labels, average='macro')
	metrics["f1"] = f1_score(gold_labels, predicted_labels, average='macro')
	metrics["acc"] = accuracy_score(gold_labels, predicted_labels)
	metrics["pool"] = final_pool_size

	return metrics

def _check_create_dict(given_dict, new_key):
	if new_key not in given_dict.keys():
		given_dict[new_key] = {}

def generate_metrics(predictions_dict):
	metrics = {}

	for set_name, set_dict in predictions_dict.iteritems():
		metrics[set_name] = {}

		for fold, fold_dict in set_dict.iteritems():

			gold_labels = fold_dict["gold_labels"]
			del fold_dict["gold_labels"]

			for hardness_type, filter_dict in fold_dict.iteritems():
				_check_create_dict(metrics[set_name], hardness_type)

				for strategy, tuple_data in filter_dict.iteritems():

					metrics_str = metrics[set_name][hardness_type]

					fold_metrics = _calculate_metrics(gold_labels, tuple_data)

					if strategy not in metrics_str.keys():
					    metrics_str[strategy] = [fold_metrics]
					else:
						metrics_str[strategy].append(fold_metrics)

	return metrics

def _summarize_metrics_folds(metrics_folds):
	summary = {}
	metric_names = metrics_folds[0].keys()

	for metric_name in metric_names:
		scores = [metrics_folds[i][metric_name] for i in xrange(len(metrics_folds))]
		summary[metric_name] = [np.mean(scores), np.std(scores)]

	return summary

def summarize_metrics_folds(metrics_dict):

	summary = deepcopy(metrics_dict)

	for set_name, set_dict in metrics_dict.iteritems():
		for hardness_type, filter_dict in set_dict.iteritems():
			for strategy, metrics_folds in filter_dict.iteritems():
				cur_summary = _summarize_metrics_folds(metrics_folds)
				summary[set_name][hardness_type][strategy] = cur_summary

	return summary

def pandanize_summary(summary):

	df = pd.DataFrame(columns = ['set', 'hardness', 'strategy',
	                  'mean_auc_roc', 'std_auc_roc', 'mean_acc', 'std_acc',
	                  'mean_f1', 'std_f1', 'mean_g1', 'std_g1',
	                  'mean_pool', 'std_pool'])

	for set_name, set_dict in summary.iteritems():
		for hardness_type, filter_dict in set_dict.iteritems():
			for strategy, summary_folds in filter_dict.iteritems():
				df_folds = pd.DataFrame(_unfilled_row(3, 10),
					                    columns = df.columns)
				_fill_dataframe_folds(df_folds, summary_folds, set_name,
					                  hardness_type, strategy)
				df = df.append(df_folds)

	return df.reset_index(drop = True)

def _unfilled_row(str_columns, float_columns):
	row = [" " for i in xrange(str_columns)]
	row.extend([0.0 for j in xrange(float_columns)])
	return [row]

def _fill_dataframe_folds(df, summary, set_name, hardness, strategy):
	df.at[0, "set"] = set_name
	df.at[0, "hardness"] = hardness
	df.at[0, "strategy"] = strategy
	return _fill_dataframe_metrics(df, summary)

def _fill_dataframe_metrics(df, summary):
	for key, metrics in summary.iteritems():
		df.at[0, "mean_" + key] = metrics[0]
		df.at[0, "std_" + key] = metrics[1]
	return df

def save_pandas_summary(df):
	pd.to_pickle(df, '../metrics/metrics_summary.pkl')

def read_pandas_summary():
	return pd.read_pickle('../metrics/metrics_summary.pkl')

def separate_pandas_summary(df, separate_sets):
	dfs = []

	if separate_sets is True:
		sets = df["set"].unique()
		for set_name in sets:
			dfs.append(df.loc[df["set"]==set_name])
	else:
		dfs.append(df)

	return dfs

def write_comparison(dfs, focus_columns, filename):

	with open('../comparisons/'+ filename + '.txt', "w") as outfile:
		for df_set in dfs:
			if len(dfs) == 1:
				outfile.write("\n\nDATASET: Mixed\n")
			else:
				outfile.write("\n\nDATASET: " + df_set.iat[0,0] + "\n")
			outfile.write("Mean of metrics\n")
			outfile.write(df_set.groupby(by=focus_columns).mean().to_string())
			outfile.write("\n\nStd of metrics\n")
			outfile.write(df_set.groupby(by=focus_columns).std().to_string())
			outfile.write("\n")
			outfile.write("-------------------------------------------------")

def bool_str(s):

    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')

    return s == 'True'