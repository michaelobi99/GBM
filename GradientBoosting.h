#pragma once
#include "DecisionTree.h"

class GradientBoosting {
private:
    std::vector<DecisionTree> trees;
    double learning_rate;
    int n_estimators;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    int random_state;
    double feature_sample_ratio;
    double initial_prediction;
    std::string loss;

public:
    GradientBoosting(int n_estimators = 100, double learning_rate = 0.1, int max_depth = 5, int min_samples_split = 4, int min_samples_leaf = 1, 
        double feature_sample_ratio = 1.0, std::string loss = "MSE")
        : n_estimators(n_estimators), learning_rate(learning_rate), max_depth(max_depth), min_samples_split(min_samples_split), min_samples_leaf(min_samples_leaf),
          feature_sample_ratio(feature_sample_ratio), initial_prediction(0.0), loss(loss), random_state(0)
    {}

    void fit(const matrix<double>& X, const std::vector<double>& Y) {
        if (X.empty() || Y.empty()) return;

        const double huber_delta = 12.0;
        const double sub_sampling_ratio = 0.8;

        initial_prediction = std::accumulate(Y.begin(), Y.end(), 0.0) / (double)Y.size();

        std::vector<double> current_predictions(Y.size(), initial_prediction);

        std::random_device rd;
        if (random_state == -1)
            random_state = rd();
        std::mt19937 rng(random_state);

        for (int i = 0; i < n_estimators; ++i) {
            std::vector<double> residuals(Y.size());

            for (size_t j = 0; j < Y.size(); ++j) {
                double error = Y[j] - current_predictions[j];
                if (loss == "HUBER") {
                    if (std::abs(error) <= huber_delta)
                        residuals[j] = error;
                    else
                        residuals[j] = huber_delta * (error > 0 ? 1.0 : -1.0);
                }
                else
                    residuals[j] = error; //use MSE loss gradient
            }

            std::vector<int> indices(X.size());
            std::iota(std::begin(indices), std::end(indices), 0);
            std::shuffle(std::begin(indices), std::end(indices), rng);

            size_t sample_size = (size_t)(sub_sampling_ratio * X.size());
            std::vector<int> sample_indices(std::begin(indices), std::begin(indices) + sample_size);

            matrix<double> X_sample(sample_size);
            std::vector<double> residual_sample(sample_size);

            for (int i = 0; i < sample_size; ++i) {
                X_sample[i] = X[sample_indices[i]];
                residual_sample[i] = residuals[sample_indices[i]];
            }

            DecisionTree tree(max_depth, min_samples_split, min_samples_leaf, feature_sample_ratio, loss);
            tree.fit(X_sample, residual_sample);
            trees.push_back(std::move(tree));

            // Update the predictions of the ensemble
            for (size_t j = 0; j < X.size(); ++j) {
                auto [pred, _] = trees.back().predict(X[j]);
                current_predictions[j] += learning_rate * pred;
            }
        }
    }

    double predict(const std::vector<double>& X) const {
        if (X.empty()) return 0;
        double result = initial_prediction;

        for (const auto& tree : trees) {
            auto [pred, _] = tree.predict(X);
            result += learning_rate * pred;
        }
        return result;
    }

    void save(const std::string& model_file) {
        std::fstream file(model_file, std::ios::out | std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not save model to file." << std::endl;
            return;
        }

        file.write(reinterpret_cast<const char*>(&n_estimators), sizeof(n_estimators));
        file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
        file.write(reinterpret_cast<const char*>(&random_state), sizeof(random_state));
        file.write(reinterpret_cast<const char*>(&initial_prediction), sizeof(initial_prediction));

        for (auto& tree : trees) {
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

        file.read(reinterpret_cast<char*>(&n_estimators), sizeof(n_estimators));
        file.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
        file.read(reinterpret_cast<char*>(&random_state), sizeof(random_state));
        file.read(reinterpret_cast<char*>(&initial_prediction), sizeof(initial_prediction));

        trees.clear();
        for (int i = 0; i < n_estimators; ++i) {
            DecisionTree tree;
            tree.load(file);
            trees.push_back(std::move(tree));
        }
        file.close();
    }
};
