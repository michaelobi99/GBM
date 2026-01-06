#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <sstream>
#include <tuple>
#include <fstream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <chrono>


template <typename T>
using matrix = std::vector<std::vector<T>>;


//helper functions
//....................................................................................................................
double evaluate_RMSE(const std::vector<double>& Y_test, const std::vector<double>& Y_pred) {
	double mse = 0;
	for (size_t i = 0; i < Y_test.size(); ++i) {
		double error = Y_test[i] - Y_pred[i];
		mse += error * error;
	}
	mse /= Y_test.size();
	return std::sqrt(mse);
}

double evaluate_Huber_RMSE(const std::vector<double>& Y_test, const std::vector<double>& Y_pred,
	double delta = 12.0)
{
	double huber_sum = 0.0;

	for (size_t i = 0; i < Y_test.size(); ++i) {
		double error = Y_test[i] - Y_pred[i];
		double abs_err = std::abs(error);

		if (abs_err <= delta) {
			huber_sum += 0.5 * error * error;
		}
		else {
			huber_sum += delta * (abs_err - 0.5 * delta);
		}
	}

	return std::sqrt(huber_sum / static_cast<double>(Y_test.size()));
}

std::tuple<matrix<double>, matrix<double>, std::vector<double>, std::vector<double>>
train_test_split(const matrix<double>& X, const std::vector<double>& Y, double test_size = 0.10, bool time_based = true, int random_state = -1) {
	std::vector<int> indices(Y.size(), 0);
	std::iota(std::begin(indices), std::end(indices), 0);
	std::random_device rd;
	int seed = random_state != -1 ? random_state : rd();
	std::mt19937 mt(seed);

	if (!time_based)
		std::shuffle(std::begin(indices), std::end(indices), mt);

	matrix<double> X_train, X_test;
	std::vector<double> Y_train, Y_test;

	size_t split = Y.size() * (1 - test_size);


	for (int i = 0; i < split; ++i) {
		Y_train.push_back(Y[indices[i]]);
		X_train.push_back(X[indices[i]]);
	}

	for (int i = split; i < Y.size(); ++i) {
		Y_test.push_back(Y[indices[i]]);
		X_test.push_back(X[indices[i]]);
	}

	return std::tuple{ X_train,  X_test, Y_train, Y_test };
}
//..........................................................................................................................


struct TreeNode {
	bool      is_leaf;
	int	      feature_index;
	double    split_value;
	double    prediction;
	std::unordered_map<int, double> feature_importance_idx;
	std::vector<double> samples;
	TreeNode* left;
	TreeNode* right;

	TreeNode() : is_leaf{ false }, feature_index{ -1 }, split_value{ -1 }, prediction{ -1 },
		left{ nullptr }, right{ nullptr } {
	}

	~TreeNode() {
		if (left)  delete left;
		if (right) delete right;
	}
};

struct DecisionTree {
private:
	int max_depth;
	int min_samples_split;
	int min_samples_leaf;
	double feature_sample_ratio;
	std::unordered_map<int, double> feature_importance_idx;
	std::string loss;
	

	double calculate_MSE(const std::vector<double>& labels) {
		double n = (double)(labels.size());
		if (n == 0) return 0.0;
		double mean = std::accumulate(std::begin(labels), std::end(labels), 0) / n;

		double mse = 0;
		for (auto elem : labels) {
			double diff = elem - mean;
			mse += diff * diff;
		}
		return mse / n;
	}

	void split_data(std::vector<int>& left_indices, std::vector<int>& right_indices, 
		const matrix<double>& X, const std::vector<int>& indices, int feature_idx, double split_value) {
		for (auto idx : indices) {
			if (X[idx][feature_idx] <= split_value) {
				left_indices.push_back(idx);
			}
			else right_indices.push_back(idx);
		}
	}

	std::tuple<int, double, double> find_best_split(const matrix<double>& X, const std::vector<double>& Y, const std::vector<int>& indices) {
		double best_mse = -1.0;
		int best_feature = -1;
		double best_value = 0;

		// Calculate parent stats once
		double sum_total = 0;
		double sum_sq_total = 0;
		for (int idx : indices) {
			sum_total += Y[idx];
			sum_sq_total += Y[idx] * Y[idx];
		}
		int N = indices.size();

		double parent_mse = (sum_sq_total - (sum_total * sum_total / (double)N));
		parent_mse /= (double)N;

		size_t number_of_features = X[0].size();
		std::random_device rd;
		std::mt19937 rng(rd());

		if (feature_sample_ratio > 1.0)
			feature_sample_ratio = 1.0;

		int nFeatures = std::max(1, (int)std::round(feature_sample_ratio * number_of_features));

		std::vector<int> feature_indices(number_of_features);
		std::iota(std::begin(feature_indices), std::end(feature_indices), 0);

		if (feature_sample_ratio != 1.0)
			std::shuffle(std::begin(feature_indices), std::end(feature_indices), rng);

		std::vector<int> chosen_features(std::begin(feature_indices), std::begin(feature_indices) + nFeatures);

		for (int feature_idx : chosen_features) {
			// 1. Sort indices based on feature values
			std::vector<std::pair<double, int>> sorted_samples;
			sorted_samples.reserve(N);
			for (int idx : indices) {
				sorted_samples.push_back({ X[idx][feature_idx], idx });
			}
			std::sort(sorted_samples.begin(), sorted_samples.end());

			// 2. Sliding Window Scan
			double sum_left = 0, sum_sq_left = 0;
			int count_left = 0;

			// Stop at N-1 because we need at least one element on the right
			for (int i = 0; i < N - 1; ++i) {
				double val = Y[sorted_samples[i].second];
				sum_left += val;
				sum_sq_left += val * val;
				count_left++;

				// Skip splits that create tiny leaves
				if (count_left < min_samples_leaf) continue;
				int count_right = N - count_left;
				if (count_right < min_samples_leaf) continue;

				// Only check split if the next value is different (handles duplicate feature values)
				if (sorted_samples[i].first == sorted_samples[i + 1].first) continue;

				// Calculate MSE for Left and Right using the variance formula:
				// MSE = (SumSq - (Sum^2 / N)) / N
				double sum_right = sum_total - sum_left;
				double sum_sq_right = sum_sq_total - sum_sq_left;

				double mse_left = (sum_sq_left - (sum_left * sum_left / count_left));
				double mse_right = (sum_sq_right - (sum_right * sum_right / count_right));

				double total_weighted_mse = (mse_left + mse_right) / N;

				if (best_feature == -1 || total_weighted_mse < best_mse) {
					best_mse = total_weighted_mse;
					best_feature = feature_idx;
					best_value = (sorted_samples[i].first + sorted_samples[i + 1].first) / 2.0;
				}
			}
		}
		double gain = parent_mse - best_mse;
		if (best_feature != -1)
			feature_importance_idx[best_feature] += gain;
		return { best_feature, best_value, best_mse };
	}


	TreeNode* build_tree(const matrix<double>& X, const std::vector<double>& Y, const std::vector<int>& indices, unsigned depth) {
		TreeNode* node = new TreeNode();

		if (indices.empty()) {
			node->is_leaf = true;
			node->prediction = 0.0;
			return node;
		}

		if (depth >= max_depth || indices.size() <= min_samples_split) {
			node->is_leaf = true;
			node->samples.clear();
			node->samples.resize(indices.size());
			
			for (auto i{ 0 }; i < indices.size(); ++i) node->samples[i] = Y[indices[i]];

			if (loss == "MSE") {
				node->prediction = mean(node->samples);
			}
			else if (loss == "HUBER") {
				node->prediction = median(node->samples);
			}
				
			else {
				std::cout << "ERROR: Unknown loss\n";
				exit(1);
			}
			return node;
		}

		auto [feature_idx, split_value, mse] = find_best_split(X, Y, indices);

		if (feature_idx == -1) {
			node->is_leaf = true;
			node->samples.clear();
			node->samples.resize(indices.size());

			for (auto i{ 0 }; i < indices.size(); ++i) node->samples[i] = Y[indices[i]];

			if (loss == "MSE") {
				node->prediction = mean(node->samples);
			}
			else if (loss == "HUBER") {
				node->prediction = median(node->samples);
			}

			else {
				std::cout << "ERROR: Unknown loss\n";
				exit(1);
			}
			return node;
		}


		std::vector<int> left_indices, right_indices;
		left_indices.reserve(indices.size()); right_indices.reserve(indices.size());
		split_data(left_indices, right_indices, X, indices, feature_idx, split_value);
		left_indices.shrink_to_fit(), right_indices.shrink_to_fit();

		if (left_indices.size() < min_samples_leaf || right_indices.size() < min_samples_leaf) {
			node->is_leaf = true;
			node->samples.clear();
			node->samples.resize(indices.size());

			for (auto i{ 0 }; i < indices.size(); ++i) node->samples[i] = Y[indices[i]];

			if (loss == "MSE") {
				node->prediction = mean(node->samples);
			}
			else if (loss == "HUBER") {
				node->prediction = median(node->samples);
			}

			else {
				std::cout << "ERROR: Unknown loss\n";
				exit(1);
			}
			return node;
		}

		node->is_leaf = false;
		node->feature_index = feature_idx;
		node->split_value = split_value;
		node->left = build_tree(X, Y, left_indices, depth + 1);
		node->right = build_tree(X, Y, right_indices, depth + 1);
		return node;
	}

	double sum(const std::vector<double>& array) {
		return std::accumulate(std::begin(array), std::end(array), 0.0);
	}

	double mean(const std::vector<double>& array) {
		if (array.empty()) return 0.0;
		return sum(array) / (double)array.size();
	}

	double median(std::vector<double> array) {
		if (array.empty()) return 0.0;
		std::sort(array.begin(), array.end());
		size_t n = array.size();
		if (n % 2 == 1)
			return array[n / 2];
		return 0.5 * (array[n / 2 - 1] + array[n / 2]);
	}

	void serialize(std::fstream& model_file, TreeNode* node) {
		// Handle nullptr nodes
		if (node == nullptr) {
			bool is_null = true;
			model_file.write(reinterpret_cast<const char*>(&is_null), sizeof(is_null));
			return;
		}
		// Mark this node as not null
		bool is_null = false;
		model_file.write(reinterpret_cast<const char*>(&is_null), sizeof(is_null));
		model_file.write(reinterpret_cast<const char*>(&node->is_leaf), sizeof(node->is_leaf));
		model_file.write(reinterpret_cast<const char*>(&node->feature_index), sizeof(node->feature_index));
		model_file.write(reinterpret_cast<const char*>(&node->split_value), sizeof(node->split_value));
		model_file.write(reinterpret_cast<const char*>(&node->prediction), sizeof(node->prediction));
		size_t size = node->samples.size();
		model_file.write(reinterpret_cast<const char*>(&size), sizeof(size_t));
		if (size > 0)
			model_file.write(reinterpret_cast<const char*>(node->samples.data()), sizeof(double) * size);
		//serialize children recursively
		if (!node->is_leaf) {
			serialize(model_file, node->left);
			serialize(model_file, node->right);
		}
	}

	TreeNode* deserialize(std::fstream& model_file) {
		bool is_null = false;
		model_file.read(reinterpret_cast<char*>(&is_null), sizeof(is_null));
		if (is_null) {
			return nullptr;
		}
		TreeNode* node = new TreeNode();
		model_file.read(reinterpret_cast<char*>(&node->is_leaf), sizeof(node->is_leaf));
		model_file.read(reinterpret_cast<char*>(&node->feature_index), sizeof(node->feature_index));
		model_file.read(reinterpret_cast<char*>(&node->split_value), sizeof(node->split_value));
		model_file.read(reinterpret_cast<char*>(&node->prediction), sizeof(node->prediction));
		size_t size = 0;
		model_file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
		if (size > 0) {
			node->samples.resize(size);
			model_file.read(reinterpret_cast<char*>(node->samples.data()), sizeof(double) * size);
		}
		//deserialize children recursively
		if (!node->is_leaf) {
			node->left = deserialize(model_file);
			node->right = deserialize(model_file);
		}
		return node;
	}

public:
	DecisionTree(int max_depth = 5, int min_samples_split = 4, int min_samples_leaf = 1, double feature_sample_ratio = 1.0, std::string loss = "MSE") :
		max_depth{ max_depth }, min_samples_split{ min_samples_split }, min_samples_leaf{ min_samples_leaf },
		feature_sample_ratio{ feature_sample_ratio }, loss{ loss }
	{}

	TreeNode* fit(const matrix<double>& X, const std::vector<double>& Y) {
		std::vector<int> indices(Y.size());
		std::iota(std::begin(indices), std::end(indices), 0);
		TreeNode* root = build_tree(X, Y, indices, 0);
		root->feature_importance_idx = feature_importance_idx;
		return root;

	}

	std::pair<double, std::vector<double>> predict(TreeNode* root, const std::vector<double>& predictors) const {
		const TreeNode* node = root;
		while (node && !node->is_leaf) {
			node = predictors[node->feature_index] <= node->split_value ? node->left : node->right;
		}
		return node ? std::pair{node->prediction, node->samples} : std::pair{0.0, std::vector<double>()};
	}

	void save(TreeNode* root, std::fstream& model_file_obj) {
		serialize(model_file_obj, root);
	}

	void load(TreeNode*& root, std::fstream& model_file_obj) {
		root = deserialize(model_file_obj);
	}
};