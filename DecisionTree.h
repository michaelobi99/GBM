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


std::vector<std::string> keynames;

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


//dataframe load_data(const std::string& filename) {
//	auto split = [](const std::string& str, char delimiter) {
//		std::vector<std::string> fields;
//		std::stringstream ss(str);
//		std::string field;
//
//		while (std::getline(ss, field, delimiter)) {
//			fields.push_back((field));
//		}
//		return fields;
//		};
//
//	std::fstream file{ filename, std::ios_base::in };
//	dataframe spreadsheet;
//	std::vector<std::string> header_names;
//
//	std::string line;
//	std::getline(file, line);
//	for (std::string key : split(line, ',')) {
//		header_names.push_back(key);
//		spreadsheet[key] = std::vector<double>{};
//	}
//
//	while (std::getline(file, line)) {
//		std::vector<std::string> row = split(line, ',');
//		for (size_t i{ 0 }; i < row.size(); ++i) {
//			double value;
//			try {
//				value = row[i].empty() ? -1 : std::stod(row[i]);
//			}
//			catch (...) {
//				value = -1;
//			}
//			spreadsheet[header_names[i]].push_back(value);
//		}
//	}
//	return spreadsheet;
//	file.close();
//}

std::tuple<matrix<double>, matrix<double>, std::vector<double>, std::vector<double>>
train_test_split(const matrix<double>& X, const std::vector<double>& Y, double test_size = 0.05, bool time_based = true, int random_state = -1) {
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
	TreeNode* root;
	int max_depth;
	int min_samples_split;
	int min_samples_leaf;
	double feature_sample_ratio;
	std::string loss;
	std::unordered_map<std::string, double> feature_importance;

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

	std::tuple<std::vector<int>, std::vector<int>> split_data(const matrix<double>& X, const std::vector<int>& indices, int feature_idx, double split_value) {
		std::vector<int> left_indices, right_indices;
		for (auto idx : indices) {
			if (X[idx][feature_idx] <= split_value) {
				left_indices.push_back(idx);
			}
			else right_indices.push_back(idx);
		}
		return std::tuple{ left_indices, right_indices };
	}

	std::tuple<int, double, double> find_best_split(const matrix<double>& X, const std::vector<double>& Y, const std::vector<int>& indices) {
		double best_mse = std::numeric_limits<double>::max();
		int best_feature = -1;
		double best_value = 0;
		std::random_device rd;
		std::mt19937 rng(rd());

		double parent_mse = calculate_MSE(Y);

		size_t number_of_features = X[0].size();

		if (feature_sample_ratio > 1.0)
			feature_sample_ratio = 1.0;

		int nFeatures = std::max(1, (int)std::round(feature_sample_ratio * number_of_features));

		std::vector<int> feature_indices(number_of_features);
		std::iota(std::begin(feature_indices), std::end(feature_indices), 0);

		if (feature_sample_ratio != 1.0)
			std::shuffle(std::begin(feature_indices), std::end(feature_indices), rng);

		std::vector<int> chosen_features(std::begin(feature_indices), std::begin(feature_indices) + nFeatures);

		for (int feature_idx : chosen_features) {
			std::vector<double> values;
			for (auto idx : indices) {
				values.push_back(X[idx][feature_idx]);
			}

			if (values.empty())
				continue;
			std::sort(values.begin(), values.end());
			auto last = std::unique(values.begin(), values.end());
			values.erase(last, values.end());

			for (double value : values) {
				auto [left_indices, right_indices] = split_data(X, indices, feature_idx, value);
				if (left_indices.size() < min_samples_leaf || right_indices.size() < min_samples_leaf) continue;

				std::vector<double> left_label, right_label;
				for (auto idx : left_indices) left_label.push_back(Y[idx]);
				for (auto idx : right_indices) right_label.push_back(Y[idx]);

				double left_mse = calculate_MSE(left_label);
				double right_mse = calculate_MSE(right_label);

				double weighted_mse = (left_label.size() * left_mse + right_label.size() * right_mse) / indices.size();

				if (weighted_mse < best_mse) {
					best_mse = weighted_mse;
					best_feature = feature_idx;
					best_value = value;
				}
			}
		}

		double gain = parent_mse - best_mse;
		if (best_feature != -1)
			feature_importance[keynames[best_feature]] += gain;
		if (best_mse < parent_mse) {
			return { best_feature, best_value, best_mse };
		}
		return { -1, 0.0, 0.0 };
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
			for (auto idx : indices) {
				node->samples.push_back(Y[idx]);
			}
			if (loss == "MSE")
				node->prediction = mean(node->samples);
			else if (loss == "HUBER")
				node->prediction = median(node->samples);
			else {
				std::cout << "ERROR: Unknown loss\n";
				exit(1);
			}
			return node;
		}

		auto [feature_idx, split_value, mse] = find_best_split(X, Y, indices);
		if (feature_idx == -1) {
			node->is_leaf = true;
			for (auto idx : indices) {
				node->samples.push_back(Y[idx]);
			}
			if (loss == "MSE")
				node->prediction = mean(node->samples);
			else if (loss == "HUBER")
				node->prediction = median(node->samples);
			else {
				std::cout << "ERROR: Unknown loss\n";
				exit(1);
			}
			return node;
		}

		auto [left_indices, right_indices] = split_data(X, indices, feature_idx, split_value);
		if (left_indices.size() < min_samples_leaf || right_indices.size() < min_samples_leaf) {
			node->is_leaf = true;
			for (auto idx : indices) {
				node->samples.push_back(Y[idx]);
			}
			if (loss == "MSE")
				node->prediction = mean(node->samples);
			else if (loss == "HUBER")
				node->prediction = median(node->samples);
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


	void copyTree(TreeNode*& toNode, TreeNode* fromNode) {
		if (!fromNode) toNode = nullptr;
		else {
			toNode = new TreeNode;
			toNode->is_leaf = fromNode->is_leaf;
			toNode->feature_index = fromNode->feature_index;
			toNode->split_value = fromNode->split_value;
			toNode->prediction = fromNode->prediction;
			copyTree(toNode->left, fromNode->left);
			copyTree(toNode->right, fromNode->right);
		}
	}

	void destroy(TreeNode*& node) {
		if (node != nullptr) {
			destroy(node->left);
			destroy(node->right);
			delete node;
			node = nullptr;
		}
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
		size_t s = node->samples.size();
		model_file.write(reinterpret_cast<const char*>(&s), sizeof(s));
		if (s > 0) {
			model_file.write(reinterpret_cast<const char*>(node->samples.data()), s * sizeof(int));
		}
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
		size_t s;
		model_file.read(reinterpret_cast<char*>(&s), sizeof(s));
		if (s > 0) {
			node->samples.resize(s);
			model_file.read(reinterpret_cast<char*>(node->samples.data()), s * sizeof(int));
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
		root{ nullptr }, max_depth{ max_depth }, min_samples_split{ min_samples_split }, min_samples_leaf{ min_samples_leaf },
		feature_sample_ratio{ feature_sample_ratio }, loss{ loss }
	{
	}

	DecisionTree(const DecisionTree& other) {
		if (!other.root)
			root = nullptr;
		else
			copyTree(root, other.root);
		max_depth = other.max_depth;
		min_samples_split = other.min_samples_split;
		min_samples_leaf = other.min_samples_leaf;
		feature_sample_ratio = other.feature_sample_ratio;
		feature_importance = other.feature_importance;
	}

	DecisionTree(DecisionTree&& other) noexcept {
		if (other.root)
			root = std::exchange(other.root, nullptr);
		else
			root = nullptr;
		std::swap(max_depth, other.max_depth);
		std::swap(min_samples_split, other.min_samples_split);
		std::swap(min_samples_leaf, other.min_samples_leaf);
		std::swap(feature_sample_ratio, other.feature_sample_ratio);
		std::swap(feature_importance, other.feature_importance);
	}

	DecisionTree& operator=(const DecisionTree& other) {
		if (!other.root)
			root = nullptr;
		else
			copyTree(root, other.root);
		max_depth = other.max_depth;
		min_samples_split = other.min_samples_split;
		min_samples_leaf = other.min_samples_leaf;
		feature_sample_ratio = other.feature_sample_ratio;
		feature_importance = other.feature_importance;
		return *this;
	}

	DecisionTree& operator=(DecisionTree&& other) noexcept {
		if (other.root)
			root = std::exchange(other.root, nullptr);
		else
			root = nullptr;
		std::swap(max_depth, other.max_depth);
		std::swap(min_samples_split, other.min_samples_split);
		std::swap(min_samples_leaf, other.min_samples_leaf);
		std::swap(feature_sample_ratio, other.feature_sample_ratio);
		std::swap(feature_importance, other.feature_importance);
		return *this;
	}

	~DecisionTree() {
		destroy(root);
	}

	void fit(const matrix<double>& X, const std::vector<double>& Y) {
		std::vector<int> indices(Y.size());
		std::iota(std::begin(indices), std::end(indices), 0);
		root = build_tree(X, Y, indices, 0);
	}

	std::unordered_map<std::string, double> get_feature_importance() const {
		return feature_importance;
	}

	std::tuple<double, std::vector<double>> predict(const std::vector<double>& predictors) const {
		const TreeNode* node = root;
		while (node && !node->is_leaf) {
			node = predictors[node->feature_index] <= node->split_value ? node->left : node->right;
		}
		return node ? std::tuple{ node->prediction, node->samples } : std::tuple{ 0.0, std::vector<double>() };
	}

	void save(std::fstream& model_file_obj) {
		model_file_obj.write(reinterpret_cast<const char*>(&max_depth), sizeof(max_depth));
		model_file_obj.write(reinterpret_cast<const char*>(&min_samples_split), sizeof(min_samples_split));
		model_file_obj.write(reinterpret_cast<const char*>(&min_samples_leaf), sizeof(min_samples_leaf));
		model_file_obj.write(reinterpret_cast<const char*>(&feature_sample_ratio), sizeof(feature_sample_ratio));
		serialize(model_file_obj, root);
	}

	void load(std::fstream& model_file_obj) {
		model_file_obj.read(reinterpret_cast<char*>(&max_depth), sizeof(max_depth));
		model_file_obj.read(reinterpret_cast<char*>(&min_samples_split), sizeof(min_samples_split));
		model_file_obj.read(reinterpret_cast<char*>(&min_samples_leaf), sizeof(min_samples_leaf));
		model_file_obj.read(reinterpret_cast<char*>(&feature_sample_ratio), sizeof(feature_sample_ratio));
		root = deserialize(model_file_obj);
	}

	void save(const std::string& model_file) {
		std::fstream file(model_file, std::ios::out | std::ios::binary);
		save(file);
		file.close();
	}

	void load(const std::string& model_file) {
		std::fstream file(model_file, std::ios::in | std::ios::binary);
		load(file);
		file.close();
	}

};