#define _CRT_SECURE_NO_WARNINGS


#include "RandomForest.h"
#include "GradientBoosting.h"
#include "XGBoost.h"
#include "DataFrame.h"
#include <tuple>
#include <set>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <thread>

const std::string WHITESPACE = " \n\r\t\f\v";

inline std::string ltrim(const std::string& s) {
	size_t start = s.find_first_not_of(WHITESPACE);
	return (start == std::string::npos) ? "" : s.substr(start);
}

inline std::string rtrim(const std::string& s) {
	size_t end = s.find_last_not_of(WHITESPACE);
	return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

inline std::string trim(const std::string& s) {
	return rtrim(ltrim(s));
}


int calculate_day_difference(const std::string& date_str) {
	int day, month, year;

	if (std::sscanf(date_str.c_str(), "%d.%d.%d", &day, &month, &year) != 3) {
		std::cerr << "Error parsing string value as date\n";
		exit(1);
	}

	// Initialize a tm struct with the parsed date
	std::tm target_tm = {};
	target_tm.tm_year = year - 1900; // tm_year is years since 1900
	target_tm.tm_mon = month - 1;    // tm_mon is 0-11
	target_tm.tm_mday = day;         // tm_mday is 1-31
	target_tm.tm_hour = 0;
	target_tm.tm_min = 0;
	target_tm.tm_sec = 0;
	target_tm.tm_isdst = -1; // Let mktime determine DST

	// Convert the target date structure to a time_t (seconds since epoch)
	std::time_t target_time = std::mktime(&target_tm);
	if (target_time == -1) {
		// mktime failed (e.g., invalid date like Feb 30th)
		return -99999;
	}

	// Get the current time and normalize it to midnight
	std::time_t now_time_raw = std::time(nullptr);
	std::tm current_tm = *std::localtime(&now_time_raw);

	// Normalize current time to midnight by resetting time fields
	current_tm.tm_hour = 0;
	current_tm.tm_min = 0;
	current_tm.tm_sec = 0;
	std::time_t now_time = std::mktime(&current_tm);

	double diff_seconds = std::difftime(now_time, target_time);

	// Convert the difference in seconds to days and round to the nearest integer
	// 86400 seconds = 24 hours
	int diff_days = static_cast<int>(std::round(diff_seconds / (60.0 * 60.0 * 24.0)));

	return diff_days;
}


std::vector<std::tuple<std::string, std::string>> get_matches() {
	std::vector<std::string> matches =  {
		std::string{ "Detroit Pistons vs New York Knicks" },
		std::string{ "Boston Celtics vs Chicago Bulls"},
		std::string{ "Toronto Raptors vs Atlanta Hawks"},
		std::string{ "Houston Rockets vs Phoenix Suns" },
		std::string{ "Oklahoma City Thunder vs Charlotte Hornets" },
		std::string{ "Philadelphia 76ers vs Denver Nuggets" },
		std::string{ "Los Angeles Clippers vs Golden State Warriors" },
		std::string{ "Portland Trail Blazers vs Utah Jazz"},
	};

	std::vector<std::tuple<std::string, std::string>> result;
	std::string word = "vs", team_1{ "" }, team_2{ "" };
	for (const auto& match : matches) {
		auto pos = std::search(std::cbegin(match), std::cend(match),
			std::cbegin(word), std::cend(word));
		if (pos != std::cend(match)) {
			team_1 = trim(std::string(match.cbegin(), pos));
			team_2 = trim(std::string(pos + word.size(), match.cend()));
			result.push_back({ team_1, team_2 });
		}
		else continue;
	}
	return result;
}


void get_lagged_predictor_values(const std::vector<std::string>& home_features, const std::vector<std::string>& away_features, 
	const std::string team, dataframe& predictor, dataframe& data, bool home_team) {
	std::ostringstream stream;
	std::vector<std::string> home_teams{ data["HOME"].get_data()};
	std::vector<std::string> away_teams{ data["AWAY"].get_data()};
	int index = 0;
	bool is_home = true;
	for (index = home_teams.size() - 1; index >= 0; --index) {
		if (home_teams[index] == team) {
			break;
		}
		else if (away_teams[index] == team) {
			is_home = false;
			break;
		}
	}
	if (is_home) {
		if (home_team) {
			//std::cout << team << ":\n";
			for (const auto& key : home_features) {
				//std::cout << key << ": ";
				if (key == "H_REST_DAYS") {
					stream << calculate_day_difference(data["DATE"][index]) + 1;
					predictor[key] = std::vector<std::string>{ stream.str() };
					stream.str("");
				}
				else
					predictor[key] = std::vector<std::string>{ data[key][index] };
				//std::cout << predictor[key][0] << "\n";
			}
		}
		else {
			//std::cout << team << ":\n";
			for (size_t i = 0; i < home_features.size(); ++i) {
				std::string key = away_features[i];
				//std::cout << key << ": ";
				if (key == "A_REST_DAYS") {
					stream << calculate_day_difference(data["DATE"][index]) + 1;
					predictor[key] = std::vector<std::string>{ stream.str() };
					stream.str("");
				}
				else
					predictor[key] = std::vector<std::string>{ data[home_features[i]][index] };
				//std::cout << predictor[key][0] << "\n";
			}
		}
		
	}
	else {
		if (home_team) {
			//std::cout << team << ":\n";
			for (size_t i = 0; i < home_features.size(); ++i) {
				std::string key = home_features[i];
				//std::cout << key << ": ";
				if (key == "H_REST_DAYS") {
					stream << calculate_day_difference(data["DATE"][index]) + 1;
					predictor[key] = std::vector<std::string>{ stream.str() };
					stream.str("");
				}
				else
					predictor[key] = std::vector<std::string>{ data[away_features[i]][index] };
				//std::cout << predictor[key][0] << "\n";
			}
		}
		else {
			//std::cout << team << ":\n";
			for (const auto& key : away_features) {
				//std::cout << key << ": ";
				if (key == "A_REST_DAYS") {
					stream << calculate_day_difference(data["DATE"][index]) + 1;
					predictor[key] = std::vector<std::string>{ stream.str() };
					stream.str("");
				}
				else
					predictor[key] = std::vector<std::string>{ data[key][index] };
				//std::cout << predictor[key][0] << "\n";
			}
		}
		
	}
}

void gbm_thread(
	matrix<double> X_train, matrix<double> X_test, std::vector<double> Y_train, std::vector<double> Y_test, std::string gbm_model_file)
{
	// GradientBoosting booster(1000, 0.01, 5, 20, 10, 0.7, HUBER);
	GradientBoosting booster(10000, 0.003, 2, 30, 25, 0.5);
	booster.fit(X_train, Y_train);

	std::vector<double> preds2;
	for (auto i{ 0u }; i < X_test.size(); ++i) {
		double val = booster.predict(X_test[i]);
		preds2.push_back(val);
		std::cout << "Real value: " << Y_test[i] << " - Booster Prediction : " << val << "\n";
	}
	//std::cout << "Gradient Boosting Huber RMSE: " << evaluate_Huber_RMSE(Y_test, preds2) << "\n";
	std::cout << "Gradient Boosting RMSE: " << evaluate_RMSE(Y_test, preds2) << "\n";
	booster.save(gbm_model_file);
}

void xgb_thread(
	matrix<double> X_train, matrix<double> X_test, std::vector<double> Y_train, std::vector<double> Y_test, std::string gbm_model_file)
{
	XGBoostRegressor model(10000, 0.003, 2, 30, 0.5, 10, 2.0);
	model.fit(X_train, Y_train);

	std::vector<double> preds3;
	for (auto i{ 0u }; i < X_test.size(); ++i) {
		double val = model.predict(X_test[i]);
		preds3.push_back(val);
		std::cout << "Real value: " << Y_test[i] << " - Booster Prediction : " << val << "\n";
	}
	std::cout << "Xtreme Gradient Boosting RMSE: " << evaluate_RMSE(Y_test, preds3) << "\n";
	model.save(gbm_model_file);
}

int main() {
	
	std::vector<std::string> predictors = {
		// "DATE", "HOME", "AWAY", "H_SCORE", "A_SCORE",
		// "H_FGA", "A_FGA", "H_FG", "A_FG", "H_FG%", "A_FG%", "H_2FGA", "A_2FGA",	"H_2FG", "A_2FG", 
		// "H_3FGA", "A_3FGA", "H_3FG", "A_3FG", 
		// "H_FTA", "A_FTA", "H_FT", "A_FT", "H_OREB", "A_OREB", "H_DREB", "A_DREB", "H_TREB", "A_TREB", "H_AST", "A_AST", "H_BLKS", "A_BLKS",
		// "H_TOV", "A_TOV", "H_STL", "A_STL", "H_P_FOULS", "A_P_FOULS",
		// "H_2FG_RATE", "A_2FG_RATE", 
		// "H_PPP", "A_PPP", 
		//"H_POSS", "A_POSS",
		//"H_EXPECTED_SCORE", "A_EXPECTED_SCORE",
		//"PACE_X_NET_RATING"
		//"H_EFG%", "A_EFG%",
		//"NET_RATING_DIFF",
		//"H_REST_DAYS", "A_REST_DAYS",

		"H_2FG%", "A_2FG%", "H_3FG%", "A_3FG%",  "H_FT%", "A_FT%",
		"H_OFF_RATING", "A_OFF_RATING", "H_DEF_RATING", "A_DEF_RATING",
		"H_FG%_ALLOWED", "A_FG%_ALLOWED", "H_2FG%_ALLOWED", "A_2FG%_ALLOWED", "H_3FG%_ALLOWED", "A_3FG%_ALLOWED", "H_TOV_ALLOWED", "A_TOV_ALLOWED",
		"H_3FG_RATE", "A_3FG_RATE", "H_FT_RATE", "A_FT_RATE",
		"H_TOV_RATE", "A_TOV_RATE", "H_OREB_RATE", "A_OREB_RATE", "H_DREB_RATE", "A_DREB_RATE",
		"GAME_PACE",
		"H_TS%", "A_TS%", "AVG_TS%",
		"H_OFF_VS_A_DEF", "A_OFF_VS_H_DEF", "H_FG%_VS_A_ALLOWED", "A_FG%_VS_H_ALLOWED", "H_2FG%_VS_A_ALLOWED", "A_2FG%_VS_H_ALLOWED", 
		"H_3FG%_VS_A_ALLOWED", "A_3FG%_VS_H_ALLOWED",
		"EXPECTED_TOTAL",
		"H_NET_RATING", "A_NET_RATING", 
		"REST_DIFF", "PACE_X_EFFICIENCY",
	};

	/*
	dataframe basketball_data = load_data(R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\data_file.csv)");

	matrix<double> X;
	std::vector<double> Y = basketball_data["TOTAL"].to_float();

	X.assign(Y.size(), std::vector<double>(predictors.size()));

	for (size_t j = 0; j < predictors.size(); ++j) {
		const auto& feature = predictors[j];
		if (basketball_data.find(feature) != basketball_data.end()) {
			const auto& feature_vec = basketball_data[feature].to_float();
			std::cout << feature << "\n";
			for (size_t i = 0; i < std::min(feature_vec.size(), Y.size()); ++i) {
				X[i][j] = feature_vec[i];
			}
		}
	}

	auto [X_train, X_test, Y_train, Y_test] = train_test_split(X, Y);

	std::string forest_model_file = R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\Forest_model.bin)";
	std::string gbm_model_file = R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\GBM_model.bin)";
	std::string xgb_model_file = R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\XGB_model.bin)";

	std::thread worker1 = std::thread(gbm_thread, X_train, X_test, Y_train, Y_test, gbm_model_file);
	std::thread worker2 = std::thread(xgb_thread, X_train, X_test, Y_train, Y_test, xgb_model_file);

	double feature_ratio = 0.3;
	//RandomForest forest(400, 19, 10, 5, feature_ratio);
	RandomForest forest(1000, 10, 25, 10, feature_ratio);
	forest.fit(X_train, Y_train);
	
	std::vector<double> preds1;
	for (auto i{ 0u }; i < X_test.size(); ++i) {
		auto val = forest.predict(X_test[i]);
		preds1.push_back(val);
		std::cout << "Real value: " << Y_test[i] << " - Forest Prediction : " << val << "\n";
	}
	
	std::cout << "Forest RMSE: " << evaluate_RMSE(Y_test, preds1) << "\n";
	forest.save(forest_model_file);
	
	std::cout << "Random Forest Feature Importance:\n";
	for (const auto& [i, importance] : forest.computeFeatureImportances()) {
		std::cout << predictors[i] << ": " << importance << "\n\n";
	}
	
	worker1.join();
	worker2.join();
	*/

	//........................................................................................................
	//Load saved models
	
	std::string forest_model_file = R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\Forest_model.bin)";
	std::string gbm_model_file = R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\GBM_model.bin)";
	std::string xgb_model_file = R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\XGB_model.bin)";
	std::string filename = R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\data_file.csv)";

	RandomForest forest;
	forest.load(forest_model_file);

	GradientBoosting booster;
	booster.load(gbm_model_file);

	XGBoostRegressor xgb;
	xgb.load(xgb_model_file);

	
	std::vector<std::string> home_features = {
		"H_FG%", "H_2FG%", "H_3FG%", "H_FT%",
		"H_OFF_RATING", "H_DEF_RATING", "H_REST_DAYS",
		"H_FG%_ALLOWED", "H_2FG%_ALLOWED", "H_3FG%_ALLOWED", "H_TOV_ALLOWED",
		"H_3FG_RATE", "H_FT_RATE",
		"H_TOV_RATE", "H_OREB_RATE", "H_DREB_RATE",
		"H_EFG%",
		"H_TS%",
		"H_POSS",
	};
	std::vector<std::string> away_features = {
		"A_FG%", "A_2FG%", "A_3FG%", "A_FT%",
		"A_OFF_RATING", "A_DEF_RATING", "A_REST_DAYS",
		"A_FG%_ALLOWED", "A_2FG%_ALLOWED", "A_3FG%_ALLOWED", "A_TOV_ALLOWED",
		"A_3FG_RATE", "A_FT_RATE",
		"A_TOV_RATE", "A_OREB_RATE", "A_DREB_RATE",
		"A_EFG%",
		"A_TS%",
		"A_POSS",
	};

	dataframe basketball_data = load_data(filename);
	dataframe predictor;
	std::vector<std::tuple<std::string, std::string>> matches = get_matches();
	std::ostringstream stream;
	stream << std::setprecision(4);

	for (const auto& [home, away] : matches) {
		get_lagged_predictor_values(home_features, away_features, home, predictor, basketball_data, true);
		get_lagged_predictor_values(home_features, away_features, away, predictor, basketball_data, false);

		predictor["GAME_PACE"] = (predictor["H_POSS"] + predictor["A_POSS"]) * 0.5;
		predictor["AVG_TS%"] = (predictor["H_TS%"] + predictor["A_TS%"]) * 0.5;
		predictor["H_OFF_VS_A_DEF"] = predictor["H_OFF_RATING"] - predictor["A_DEF_RATING"];
		predictor["A_OFF_VS_H_DEF"] = predictor["A_OFF_RATING"] - predictor["H_DEF_RATING"];
		predictor["H_FG%_VS_A_ALLOWED"] = predictor["H_FG%"] - predictor["A_FG%_ALLOWED"];
		predictor["A_FG%_VS_H_ALLOWED"] = predictor["A_FG%"] - predictor["H_FG%_ALLOWED"];
		predictor["H_2FG%_VS_A_ALLOWED"] = predictor["H_2FG%"] - predictor["A_2FG%_ALLOWED"];
		predictor["A_2FG%_VS_H_ALLOWED"] = predictor["A_2FG%"] - predictor["H_2FG%_ALLOWED"];
		predictor["H_3FG%_VS_A_ALLOWED"] = predictor["H_3FG%"] - predictor["A_3FG%_ALLOWED"];
		predictor["A_3FG%_VS_H_ALLOWED"] = predictor["A_3FG%"] - predictor["H_3FG%_ALLOWED"];
		predictor["H_EXPECTED_SCORE"] = (predictor["H_OFF_RATING"] + predictor["A_DEF_RATING"]) * 0.5;
		predictor["A_EXPECTED_SCORE"] = (predictor["A_OFF_RATING"] + predictor["H_DEF_RATING"]) * 0.5;
		predictor["EXPECTED_TOTAL"] = predictor["H_EXPECTED_SCORE"] + predictor["A_EXPECTED_SCORE"];
		predictor["H_NET_RATING"] = predictor["H_OFF_RATING"] - predictor["H_DEF_RATING"];
		predictor["A_NET_RATING"] = predictor["A_OFF_RATING"] - predictor["A_DEF_RATING"];
		predictor["NET_RATING_DIFF"] = predictor["H_NET_RATING"] - predictor["A_NET_RATING"];
		predictor["REST_DIFF"] = predictor["H_REST_DAYS"] - predictor["A_REST_DAYS"];
		predictor["PACE_X_NET_RATING"] = predictor["GAME_PACE"] * predictor["NET_RATING_DIFF"];
		predictor["PACE_X_EFFICIENCY"] = predictor["GAME_PACE"] * predictor["AVG_TS%"];


		matrix<double> X;
		X.assign(1, std::vector<double>(predictor.size()));
		for (int i = 0; i < predictors.size(); ++i) {
			X[0][i] = predictor[predictors[i]].to_float()[0];
		}

		std::cout << home << " VS " << away << ": \n";
		//auto [pred1, prob_220, prob_230, prob_240, prob_250] = forest.predict(X[0]);
		auto pred1 = forest.predict(X[0]);
		auto pred2 = booster.predict(X[0]);
		auto pred3 = xgb.predict(X[0]);

		// stream.str("");
		// stream << prob_220 * 100.0 << "%";
		// std::string over_220_str = stream.str();
		// stream.str("");
		// stream << prob_230 * 100.0 << "%";
		// std::string over_230_str = stream.str();
		// stream.str("");
		// stream << prob_240 * 100.0 << "%";
		// std::string over_240_str = stream.str();
		// stream.str("");
		// stream << prob_250 * 100.0 << "%";
		// std::string over_250_str = stream.str();
		// 
		// stream.str("");
		// stream << 1.0 / prob_220;
		// std::string over_220_odds = stream.str();
		// stream.str("");
		// stream << 1.0 / prob_230;
		// std::string over_230_odds = stream.str();
		// stream.str("");
		// stream << 1.0 / prob_240;
		// std::string over_240_odds = stream.str();
		// stream.str("");
		// stream << 1.0 / prob_250;
		// std::string over_250_odds = stream.str();

		std::cout << "Forest prediction: " << pred1 << "\n";
		std::cout << "GBM prediction: " << pred2 << "\n";
		std::cout << "XGB prediction: " << pred3 << "\n\n";

		//	std::string lines(45, '-');
		//	std::cout << std::left << std::setw(15) << "Score Range" << std::left << std::setw(15) << "Probability" << "Odds\n";
		//	std::cout << lines << "\n";

		//	// Use descriptive variables and print the odds column correctly
		//	stream << std::setprecision(3);
		//	std::cout << std::left << std::setw(15) << "[over 220.5]" << std::left << std::setw(15) << over_220_str << over_220_odds << "\n";
		//	std::cout << std::left << std::setw(15) << "[over 230.5]" << std::left << std::setw(15) << over_230_str << over_230_odds << "\n";
		//	std::cout << std::left << std::setw(15) << "[over 240.5]" << std::left << std::setw(15) << over_240_str << over_240_odds << "\n";
		//	std::cout << std::left << std::setw(15) << "[over 250.5]" << std::left << std::setw(15) << over_250_str << over_250_odds << "\n\n";
		//
	}
	//.......................................................................................................

}












/*
	std::vector<std::vector<double>> X;
	std::vector<double> y;

	std::mt19937 rng(42);
	std::uniform_real_distribution<double> noise(-0.2, 0.2);

	for (int i = 1; i <= 2000; ++i) {
		X.push_back({ static_cast<double>(i) });

		double base_y = 2.0 * i;
		y.push_back(base_y + (i % 2 == 0 ? 0.1 : -0.1) + noise(rng));
	}

	XGBoostRegressor model(4000);
	model.fit(X, y);

	std::cout << "Prediction for x=9: " << model.predict({ 2001 }) << std::endl;
	*/