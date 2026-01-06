#pragma once

#include "DecisionTree.h"

struct featureImportance {
	int index;
	double weight;
};

class RandomForest {
private:
	std::vector<TreeNode*> trees;
	int max_depth;
	int min_samples_split;
	int min_samples_leaf;
	int nTrees;
	int random_state;
	double feature_sample_ratio;

	std::vector<int> create_bootstrap_sample(unsigned size, std::mt19937& rng) {
		std::vector<int> sample(size);
		std::uniform_int_distribution<int> dist(0, size - 1);
		for (int i = 0; i < size; ++i) {
			sample[i] = dist(rng);
		}
		return sample;
	}

public:
	RandomForest(int nTrees = 100, int max_depth = 5, int min_samples_split = 2, int min_samples_leaf = 1, double feature_sample_ratio = 1.0, int random_state = -1) : 
		max_depth(max_depth), min_samples_split(min_samples_split), min_samples_leaf(min_samples_leaf), 
		feature_sample_ratio(feature_sample_ratio), nTrees(nTrees), random_state( random_state )
	{}
		

	void fit(const matrix<double>& X, const std::vector<double>& Y) {
		std::random_device rd;
		
		if (random_state == -1)
			random_state = rd();

		std::mt19937 rng(random_state);
		
		for (int i = 0; i < nTrees; i++) {
			auto sampleIndices = create_bootstrap_sample(Y.size(), rng);

			matrix<double> sample_X;
			std::vector<double> sample_Y;
			for (int idx : sampleIndices) {
				sample_X.push_back(X[idx]);
				sample_Y.push_back(Y[idx]);
			}

			DecisionTree tree(max_depth, min_samples_split, min_samples_leaf, feature_sample_ratio);
			TreeNode* root = tree.fit(sample_X, sample_Y);
			trees.push_back(root);
		}
	}

	double predict(const std::vector<double>& predictors) const {
		if (trees.size() == 0) return 0.0;
			//return std::tuple{ 0.0, 0.0, 0.0, 0.0, 0.0 };

		std::vector<double> predictions(trees.size());
		std::vector<double> all_samples;
		int i = 0;

		DecisionTree tree;
		for (auto* root : trees) {
			auto pred = tree.predict(root, predictors);
			predictions[i++] = pred;
			/*for (const auto elem : sample_indices) {
				all_samples.push_back(elem);
			}*/
		}
		double avg = std::accumulate(std::begin(predictions), std::end(predictions), 0.0);
		avg /= (double)trees.size();
		return avg;

		/*if (all_samples.empty()) {
			return std::tuple{ avg, 0.0, 0.0, 0.0, 0.0 };
		}
		int over_220_count{ 0 }, over_230_count{ 0 }, over_240_count{ 0 }, over_250_count{ 0 };

		for (auto elem : all_samples) {
			if (elem >= 220) ++over_220_count;
			if (elem >= 230) ++over_230_count;
			if (elem >= 240) ++over_240_count;
			if (elem >= 250) ++over_250_count;
		}
		double total_samples = (double)all_samples.size();
		double prob_over_220 = over_220_count / total_samples;
		double prob_over_230 = over_230_count / total_samples;
		double prob_over_240 = over_240_count / total_samples;
		double prob_over_250 = over_250_count / total_samples;

		return std::tuple{ avg, prob_over_220, prob_over_230, prob_over_240, prob_over_250 };*/
	}
 
	std::vector<featureImportance> computeFeatureImportances() {
		std::unordered_map<int, double> importance_map;

		DecisionTree tree;
		for (const auto& root : trees) {
			for (auto& [feature, score] : root->feature_importance_idx) {
				importance_map[feature] += score;
			}
		}
		std::vector<featureImportance> features;
		double sum = 0.0;

		for (int i = 0; i < importance_map.size(); ++i) {
			double weight = 0;
			if (importance_map.contains(i)) {
				weight = importance_map.count(i) ? importance_map.at(i) : 0.0;

			}
			features.push_back({ i, weight });
			sum += weight;
		}

		if (sum > 0.0) {
			for (auto& f : features) f.weight /= sum;
		}

		std::sort(std::begin(features), std::end(features),
			[](const featureImportance& a, const featureImportance& b) {
				return a.weight > b.weight;
			});

		return features;
	}

	void save(const std::string& model_file) {
		std::fstream file(model_file, std::ios::out | std::ios::binary);
		if (!file.is_open()) {
			std::cerr << "Error: Could not save model to file." << std::endl;
			return;
		}
		file.write(reinterpret_cast<const char*>(&nTrees), sizeof(nTrees));

		DecisionTree tree;
		for (auto* root : trees) {
			tree.save(root, file);
		}
		file.close();
	}

	void load(const std::string& model_file) {
		std::fstream file(model_file, std::ios::in | std::ios::binary);
		if (!file.is_open()) {
			std::cerr << "Error: Could not load model from file." << std::endl;
			return;
		}

		for (auto* tree : trees) if (tree) delete tree;
		trees.clear();

		file.read(reinterpret_cast<char*>(&nTrees), sizeof(nTrees));

		DecisionTree tree;
		if (nTrees > 0) {
			trees.reserve(nTrees);
			for (int i{ 0 }; i < nTrees; ++i) {
				TreeNode* root = nullptr;
				tree.load(root, file);
				trees.push_back(root);
			}
		}
		file.close();
	}
};