#pragma once

#include "DecisionTree.h"

struct featureImportance {
	int index;
	double weight;
};

class RandomForest {
private:
	std::vector<DecisionTree> trees;
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
			tree.fit(sample_X, sample_Y);
			trees.push_back(std::move(tree));
		}
	}

	std::tuple<double, double, double, double, double> predict(const std::vector<double>& predictors) const {
		if (trees.size() == 0)
			return std::tuple{ 0.0, 0.0, 0.0, 0.0, 0.0 };

		std::vector<double> predictions(trees.size());
		std::vector<double> all_samples;
		int i = 0;

		for (const DecisionTree& tree : trees) {
			auto [pred, sample_indices] = tree.predict(predictors);
			predictions[i++] = pred;
			for (const auto elem : sample_indices) {
				all_samples.push_back(elem);
			}
		}
		double avg = std::accumulate(std::begin(predictions), std::end(predictions), 0.0);
		avg /= (double)trees.size();

		if (all_samples.empty()) {
			return std::tuple{ avg, 0.0, 0.0, 0.0, 0.0 };
		}
		int over_210_count{ 0 }, over_220_count{ 0 }, over_230_count{ 0 }, over_240_count{ 0 };

		for (auto elem : all_samples) {
			if (elem >= 210) ++over_210_count;
			if (elem >= 220) ++over_220_count;
			if (elem >= 230) ++over_230_count;
			if (elem >= 240) ++over_240_count;
		}
		double total_samples = (double)all_samples.size();
		double prob_over_210 = over_210_count / total_samples;
		double prob_over_220 = over_220_count / total_samples;
		double prob_over_230 = over_230_count / total_samples;
		double prob_over_240 = over_240_count / total_samples;

		return std::tuple{ avg, prob_over_210, prob_over_220, prob_over_230, prob_over_240 };
	}
 
	std::vector<featureImportance> computeFeatureImportances() {
		std::unordered_map<int, double> importance_map;

		for (const auto& tree : trees) {
			auto imp = tree.get_feature_importance();
			for (auto& [feature, score] : imp) {
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
		file.write(reinterpret_cast<const char*>(&random_state), sizeof(random_state));
		for (DecisionTree& tree : trees) {
			tree.save(file);
		}
		file.close();
	}

	void load(const std::string& model_file) {
		std::fstream file(model_file, std::ios::in | std::ios::binary);
		if (!file.is_open()) {
			std::cerr << "Error: Could not load model from file." << std::endl;
			return;
		}

		file.read(reinterpret_cast<char*>(&nTrees), sizeof(nTrees));
		file.read(reinterpret_cast<char*>(&random_state), sizeof(random_state));

		trees.clear();

		if (nTrees > 0) {
			trees.reserve(nTrees);
			for (int i{ 0 }; i < nTrees; ++i) {
				DecisionTree tree;
				tree.load(file);
				trees.push_back(std::move(tree));
			}
		}
		file.close();
	}
};