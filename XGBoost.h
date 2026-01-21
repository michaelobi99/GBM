#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <random>
#include <string>

template <typename T>
using Matrix = std::vector<std::vector<T>>;

struct Node {
    bool leaf = false;
    int feature = -1;
    double threshold = 0;
    double value = 0;
    Node* left = nullptr;
    Node* right = nullptr;

    Node() = default;
    // Recursive cleanup to prevent memory leaks
    ~Node() {
        if (left)  delete left;
        if (right) delete right;
    }
};

class XGBoostTree {
private:
    int max_depth;
    int min_leaf;
    double eta;
    double lambda;
    double gamma;
    double feature_sample_ratio;

    void serialize(std::fstream& model_file, Node* node) {
        if (node == nullptr) {
            bool is_null = true;
            model_file.write(reinterpret_cast<const char*>(&is_null), sizeof(is_null));
            return;
        }
        bool is_null = false;
        model_file.write(reinterpret_cast<const char*>(&is_null), sizeof(is_null));
        model_file.write(reinterpret_cast<const char*>(&node->leaf), sizeof(node->leaf));
        model_file.write(reinterpret_cast<const char*>(&node->feature), sizeof(node->feature));
        model_file.write(reinterpret_cast<const char*>(&node->threshold), sizeof(node->threshold));
        model_file.write(reinterpret_cast<const char*>(&node->value), sizeof(node->value));

        if (!node->leaf) {
            serialize(model_file, node->left);
            serialize(model_file, node->right);
        }
    }

    Node* deserialize(std::fstream& model_file) {
        bool is_null = false;
        if (!model_file.read(reinterpret_cast<char*>(&is_null), sizeof(is_null))) return nullptr;
        if (is_null) return nullptr;

        Node* node = new Node();
        model_file.read(reinterpret_cast<char*>(&node->leaf), sizeof(node->leaf));
        model_file.read(reinterpret_cast<char*>(&node->feature), sizeof(node->feature));
        model_file.read(reinterpret_cast<char*>(&node->threshold), sizeof(node->threshold));
        model_file.read(reinterpret_cast<char*>(&node->value), sizeof(node->value));

        if (!node->leaf) {
            node->left = deserialize(model_file);
            node->right = deserialize(model_file);
        }
        return node;
    }

public:
    XGBoostTree(int max_depth = 3, int min_leaf = 1, double eta = 0.3, double lambda = 1.0, double gamma = 0.0,
        double feature_sample_ratio = 1.0)
        : max_depth(max_depth), min_leaf(min_leaf), eta(eta), lambda(lambda), gamma(gamma), feature_sample_ratio(feature_sample_ratio) {
    }

    Node* build(const Matrix<double>& X, const std::vector<double>& g, const std::vector<double>& h, const std::vector<int>& rows, int depth) {
        if (rows.empty()) return nullptr;

        double G = 0, H = 0;
        for (int r : rows) {
            G += g[r];
            H += h[r];
        }

        Node* node = new Node();
        if (depth >= max_depth || (int)rows.size() <= min_leaf) {
            node->leaf = true;
            node->value = (-G / (H + lambda)) * eta;
            return node;
        }

        int best_feature = -1;
        double best_threshold = 0.0;
        double best_gain = 0.0;

        size_t total_features = X[0].size();
        int nFeatures = std::max(1, (int)std::round(feature_sample_ratio * total_features));

        std::vector<int> feature_indices(total_features);
        std::iota(feature_indices.begin(), feature_indices.end(), 0);

        static std::mt19937 rng(1337);
        if (feature_sample_ratio < 1.0) {
            std::shuffle(feature_indices.begin(), feature_indices.end(), rng);
        }

        for (int i = 0; i < nFeatures; ++i) {
            int f = feature_indices[i];
            std::vector<std::pair<double, int>> sorted;
            for (int r : rows) sorted.push_back({ X[r][f], r });
            std::sort(sorted.begin(), sorted.end());

            double GL = 0, HL = 0;
            for (size_t j = 0; j < sorted.size() - 1; j++) {
                int idx = sorted[j].second;
                GL += g[idx];
                HL += h[idx];

                double GR = G - GL;
                double HR = H - HL;

                double gain = 0.5 * (
                    (GL * GL) / (HL + lambda) +
                    (GR * GR) / (HR + lambda) -
                    (G * G) / (H + lambda)
                    ) - gamma;

                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = f;
                    best_threshold = (sorted[j].first + sorted[j + 1].first) / 2.0;
                }
            }
        }

        if (best_feature == -1) {
            node->leaf = true;
            node->value = (-G / (H + lambda)) * eta;
            return node;
        }

        node->feature = best_feature;
        node->threshold = best_threshold;
        std::vector<int> left_idx, right_idx;

        for (int r : rows) {
            if (X[r][best_feature] <= best_threshold) left_idx.push_back(r);
            else right_idx.push_back(r);
        }

        node->left = build(X, g, h, left_idx, depth + 1);
        node->right = build(X, g, h, right_idx, depth + 1);
        return node;
    }

    double predict_row(Node* node, const std::vector<double>& row) const {
        if (!node) return 0.0;
        if (node->leaf) return node->value;
        if (node->feature >= (int)row.size()) return node->value; // Safety
        if (row[node->feature] <= node->threshold)
            return predict_row(node->left, row);
        return predict_row(node->right, row);
    }

    void save(Node* tree, std::fstream& file) {
        serialize(file, tree);
    }

    void load(Node*& tree, std::fstream& file) {
        tree = deserialize(file);
    }
};

class XGBoostRegressor {
private:
    int n_estimators;
    int max_depth;
    int min_leaf;
    double eta;
    double lambda;
    double gamma;
    double base_score;
    double feature_sample_ratio;
    int random_state;
    std::vector<Node*> trees;

public:
    XGBoostRegressor(int n_estimators = 100, double eta = 0.1, int max_depth = 5, int min_leaf = 1,
        double feature_sample_ratio = 1.0,  double lambda = 1.0, double gamma = 0.0, int random_state = -1, double base_score = 0.5)
        : n_estimators(n_estimators), eta(eta), max_depth(max_depth), min_leaf(min_leaf),
        lambda(lambda), gamma(gamma), base_score(base_score), feature_sample_ratio(feature_sample_ratio), random_state{ random_state } {
    }

    XGBoostRegressor(const XGBoostRegressor& other) {
        this->n_estimators = other.n_estimators;
        this->max_depth = other.max_depth;
        this->min_leaf = other.min_leaf;
        this->eta = other.eta;
        this->lambda = other.lambda;
        this->gamma = other.gamma;
        this->base_score = other.base_score;
        this->feature_sample_ratio = other.feature_sample_ratio;
        this->random_state = other.random_state;
        this->trees.clear();
        this->trees.resize(other.trees.size());
        std::copy(other.trees.begin(), other.trees.end(), this->trees.begin());
    }

    ~XGBoostRegressor() {
        for (auto* tree : trees) delete tree;
    }

    void fit(const Matrix<double>& X, const std::vector<double>& y) {
        if (X.empty() || y.empty() || X.size() != y.size()) {
            std::cerr << "Error: Input size mismatch or empty data." << std::endl;
            return;
        }

        int n = (int)y.size();
        double subsampling_ratio = 1.0;

        std::vector<double> current_preds(n, base_score);

        std::random_device rd;
        if (random_state == -1)
            random_state = rd();
        std::mt19937 rng(random_state);

        for (int i = 0; i < n_estimators; ++i) {
            std::vector<double> g(n), h(n);
            for (int j = 0; j < n; ++j) {
                g[j] = current_preds[j] - y[j];
                h[j] = 1.0;
            }

            std::vector<int> indices(n);
            std::iota(std::begin(indices), std::end(indices), 0);
            std::shuffle(std::begin(indices), std::end(indices), rng);

            size_t sample_size = (size_t)(subsampling_ratio * X.size());
            std::vector<int> sample_indices(std::begin(indices), std::begin(indices) + sample_size);

            Matrix<double> X_sample(sample_size);
            std::vector<double> g_sample(sample_size), h_sample(sample_size);

            for (int i = 0; i < sample_size; ++i) {
                X_sample[i] = X[sample_indices[i]];
                g_sample[i] = g[sample_indices[i]];
                h_sample[i] = h[sample_indices[i]];
            }

            XGBoostTree tree_builder(max_depth, min_leaf, eta, lambda, gamma, feature_sample_ratio);
            std::vector<int> rows(sample_size);
            std::iota(rows.begin(), rows.end(), 0);
            Node* root = tree_builder.build(X, g, h, rows, 0);

            if (root) {
                trees.push_back(root);
                for (int k = 0; k < n; ++k) {
                    current_preds[k] += tree_builder.predict_row(root, X[k]);
                }
            }
        }
    }

    double predict(const std::vector<double>& X_row) const {
        double pred = base_score;
        XGBoostTree t_helper;
        for (auto* tree : trees) {
            pred += t_helper.predict_row(tree, X_row);
        }
        return pred;
    }

    void save(const std::string& filename) {
        std::fstream file(filename, std::ios::out | std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not save model to file." << std::endl;
            return;
        }
        int tree_count = trees.size();
        file.write(reinterpret_cast<const char*>(&tree_count), sizeof(tree_count));
        file.write(reinterpret_cast<const char*>(&base_score), sizeof(base_score));

        XGBoostTree t_helper;
        for (auto* tree : trees) {
            t_helper.save(tree, file);
        }
    }

    void load(const std::string& filename) {
        std::fstream file(filename, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Could not load model from file." << std::endl;
            return;
        }

        for (auto* tree : trees) if (tree) delete tree;
        trees.clear();

        int tree_count;
        file.read(reinterpret_cast<char*>(&tree_count), sizeof(tree_count));
        file.read(reinterpret_cast<char*>(&base_score), sizeof(base_score));

        XGBoostTree t_helper;
        trees.reserve(tree_count);
        for (int i = 0; i < tree_count; ++i) {
            Node* tree = nullptr;
            t_helper.load(tree, file);
            if (tree) trees.push_back(tree);
        }
        file.close();
    }
};