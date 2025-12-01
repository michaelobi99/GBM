#define _CRT_SECURE_NO_WARNINGS


#include "RandomForest.h"
#include "GradientBoosting.h"
#include <tuple>
#include <set>
#include <ctime>
#include <cmath>
#include <cstdio> // For sscanf
#include <thread>


int calculate_day_difference(const std::string& date_str) {
	int day, month, year;

	// Use sscanf to safely parse the fixed DD.MM.YYYY format
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
	return std::vector<std::tuple<std::string, std::string>>{
			std::tuple{ "Utah Jazz", "Houston Rockets"},
			std::tuple{ "Cleveland Cavaliers", "Boston Celtics" },
			std::tuple{ "New York Knicks", "Toronto Raptors" },
			std::tuple{ "Philadelphia 76ers", "Atlanta Hawks" },
			std::tuple{ "Portland Trail Blazers", "Oklahoma City Thunder" },
			std::tuple{ "Minnesota Timberwolves", "San Antonio Spurs" },
			std::tuple{ "Sacramento Kings", "Memphis Grizzlies" },
			std::tuple{ "Los Angeles Lakers", "New Orleans Pelicans" },
			/*std::tuple{"Los Angeles Clippers", "Dallas Mavericks"},
			std::tuple{ "Utah Jazz", "Sacramento Kings" },
			std::tuple{ "Los Angeles Lakers", "Dallas Mavericks" },
			std::tuple{ "Los Angeles Clippers", "Memphis Grizzlies" },*/
	};
}

void get_teams_lagged_feature(std::unordered_map<std::string, std::vector<double>>& teams_lagged_features, const std::string& filename, 
	const std::set<std::string>& teams, std::unordered_map<std::string, std::string>& last_match_day) {
	std::fstream file{ filename, std::ios::in };
	if (!file.is_open()) {
		std::cout << "Error opening file...\n";
		exit(1);
	}
	std::string line;
	std::getline(file, line);
	//TODO: present game rest day is not accurate. It uses the last game rest days. Correct this
	

	while (std::getline(file, line)) {
		std::istringstream stream(line);
		std::string col;
		std::string date, home, away;
		std::vector<double> values;
		std::getline(stream, col, ',');//date
		date = col;
		std::getline(stream, col, ',');//home
		home = col;
		std::getline(stream, col, ',');//away
		away = col;
		while (std::getline(stream, col, ',')) {
			values.push_back(std::stod(col));
		}
		values = std::vector<double>(values.begin() + 2, std::prev(values.end()));
		int home_idx{ -1 }, away_idx{ -1 };
		if (teams.find(home) != teams.end()) {
			last_match_day[home] = date;
			std::vector<double> temp;
			temp.reserve(26);
			for (int i = 0; i < values.size(); i += 2) {
				temp.push_back(values[i]);
			}
			teams_lagged_features[home] = temp;
		}
		if (teams.find(away) != teams.end()) {
			last_match_day[away] = date;
			std::vector<double> temp;
			temp.reserve(26);
			for (int i = 1; i < values.size(); i += 2) {
				temp.push_back(values[i]);
			}
			teams_lagged_features[away] = temp;
		}
	}
}

void gbm_thread(
	matrix<double> X_train, matrix<double> X_test, std::vector<double> Y_train, std::vector<double> Y_test,
	std::vector<std::tuple<std::string, std::string>> matches, std::unordered_map<std::string, std::vector<double>> teams_lagged_features,
	std::unordered_map<std::string, std::string> last_match_day, std::string gbm_model_file) 
{
	GradientBoosting booster(900, 0.01, 7, 2, 1, 1.0, "HUBER");
	booster.fit(X_train, Y_train);

	std::vector<double> preds2;
	for (auto i{ 0u }; i < X_test.size(); ++i) {
		double val = booster.predict(X_test[i]);
		preds2.push_back(val);
		std::cout << "Real value: " << Y_test[i] << " - Booster Prediction : " << val << "\n";
	}
	std::cout << "Gradient Boosting Huber RMSE: " << evaluate_Huber_RMSE(Y_test, preds2) << "\n";
	booster.save(gbm_model_file);

	for (auto& match : matches) {
		auto& [home, away] = match;
		std::vector<double> predictors;
		std::vector<double> vec1 = teams_lagged_features[home];
		std::vector<double> vec2 = teams_lagged_features[away];


		vec1.back() = (double)(calculate_day_difference(last_match_day[home]) + 1);
		vec2.back() = (double)(calculate_day_difference(last_match_day[away]) + 1);


		for (int i = 0; i < vec1.size(); ++i) {
			predictors.push_back(vec1[i]);
			predictors.push_back(vec2[i]);
		}

		std::cout << home << " VS " << away << ": ";
		auto pred2 = booster.predict(predictors);
		std::cout << pred2 << "\n\n";
	}
}

int main() {
	dataframe basketball_data = load_data(R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\combined_lagged_averages.csv)");

	matrix<double> X;
	std::vector<double> Y = basketball_data["TOTAL"];

	std::vector<std::string> predictors = { 
		"H_FGA", "A_FGA", "H_FG", "A_FG", "H_FG%", "A_FG%", "H_2FGA", "A_2FGA",	"H_2FG", "A_2FG", "H_2FG%", "A_2FG%", "H_3FGA", "A_3FGA", "H_3FG", "A_3FG", "H_3FG%",
		"A_3FG%", "H_FTA", "A_FTA", "H_FT", "A_FT", "H_FT%", "A_FT%", "H_OREB", "A_OREB", "H_DREB", "A_DREB", "H_TREB", "A_TREB", "H_AST", "A_AST", "H_BLKS", "A_BLKS",	
		"H_TOV", "A_TOV", "H_STL", "A_STL", "H_P_FOULS", "A_P_FOULS", "H_OFF_RATING", "A_OFF_RATING", "H_DEF_RATING", "A_DEF_RATING", "H_REST_DAYS", "A_REST_DAYS" };

	X.assign(Y.size(), std::vector<double>(predictors.size()));

	for (size_t j = 0; j < predictors.size(); ++j) {
		const auto& feature = predictors[j];
		const auto& feature_vec = basketball_data[feature];

		for (size_t i = 0; i < std::min(feature_vec.size(), Y.size()); ++i) {
			X[i][j] = feature_vec[i];
		}
	}

	auto [X_train, X_test, Y_train, Y_test] = train_test_split(X, Y);


	std::string forest_model_file = R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\Forest_model.bin)";
	std::string gbm_model_file = R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\GBM_model.bin)";
	std::string filename = R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\combined_lagged_averages.csv)";

	std::vector<std::tuple<std::string, std::string>> matches = get_matches();
	
	std::unordered_map<std::string, std::vector<double>> teams_lagged_features;
	std::set<std::string> teams;

	for (auto& match : matches) {
		auto& [home, away] = match;
		teams.insert(home);
		teams.insert(away);
	}

	std::unordered_map<std::string, std::string> last_match_day;
	get_teams_lagged_feature(teams_lagged_features, filename, teams, last_match_day);

	
	std::thread worker = std::thread(gbm_thread, X_train, X_test, Y_train, Y_test, matches, teams_lagged_features, last_match_day, gbm_model_file);

	double feature_ratio = std::pow(predictors.size(), 0.3);
	RandomForest forest(500, 25, 2, 1, feature_ratio);
	forest.fit(X_train, Y_train);

	std::vector<double> preds1;
	for (auto i{ 0u }; i < X_test.size(); ++i) {
		auto [val, lower, upper] = forest.predict(X_test[i]);
		preds1.push_back(val);
		std::cout << "Real value: " << Y_test[i] << " - Forest Prediction : " << val << "[" << lower << " - " << upper << "]\n";
	}

	std::cout << "Forest RMSE: " << evaluate_RMSE(Y_test, preds1) << "\n";
	forest.save(forest_model_file);

	std::cout << "Randon Forest Feature Importance:\n";
	for (const auto& [feature, importance] : forest.computeFeatureImportances()) {
		std::cout << feature << ": " << importance << "\n\n";
	}

	
	for (auto& match : matches) {
		auto& [home, away] = match;
		std::vector<double> predictors;
		std::vector<double> vec1 = teams_lagged_features[home];
		std::vector<double> vec2 = teams_lagged_features[away];


		vec1.back() = (double)(calculate_day_difference(last_match_day[home]) + 1);
		vec2.back() = (double)(calculate_day_difference(last_match_day[away]) + 1);

		std::cout << std::format("{} last played {} day(s) ago\n", home, vec1.back());
		std::cout << std::format("{} last played {} day(s) ago\n", away, vec2.back());


		for (int i = 0; i < vec1.size(); ++i) {
			predictors.push_back(vec1[i]);
			predictors.push_back(vec2[i]);
		}

		std::cout << home << " VS " << away <<": ";
		auto [pred, lower, upper] = forest.predict(predictors);
		std::cout << pred << " [" << lower << "-" << upper << "]\n\n";
	}

	worker.join();
	


	//........................................................................................................
	//Load saved model
	

	/*std::string forest_model_file = R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\Forest_model.bin)";
	std::string gbm_model_file = R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\GBM_model.bin)";
	std::string filename = R"(C:\Users\HP\source\repos\Rehoboam\Rehoboam\Data\Basketball\nba\combined_lagged_averages.csv)";

	std::vector<std::tuple<std::string, std::string>> matches = get_matches();

	std::unordered_map<std::string, std::vector<double>> teams_lagged_features;
	std::set<std::string> teams;

	for (auto& match : matches) {
		auto& [home, away] = match;
		teams.insert(home);
		teams.insert(away);
	}

	std::unordered_map<std::string, std::string> last_match_day;
	get_teams_lagged_feature(teams_lagged_features, filename, teams, last_match_day);


	RandomForest forest;
	forest.load(forest_model_file);

	GradientBoosting booster;
	booster.load(gbm_model_file);

	for (auto& match : matches) {
		auto& [home, away] = match;
		std::vector<double> predictors;
		std::vector<double> vec1 = teams_lagged_features[home];
		std::vector<double> vec2 = teams_lagged_features[away];


		vec1.back() = (double)(calculate_day_difference(last_match_day[home]) + 1);
		vec2.back() = (double)(calculate_day_difference(last_match_day[away]) + 1);

		//std::cout << std::format("{} last played {} day(s) ago\n", home, vec1.back());
		//std::cout << std::format("{} last played {} day(s) ago\n", away, vec2.back());


		for (int i = 0; i < vec1.size(); ++i) {
			predictors.push_back(vec1[i]);
			predictors.push_back(vec2[i]);
		}

		std::cout << home << " VS " << away << ": \n";
		auto [pred, lower, upper] = forest.predict(predictors);
		auto pred2 = booster.predict(predictors);

		std::cout << "forest prediction: " << pred << " [" << lower << "-" << upper << "]\n";
		std::cout << "GBM prediction: " << pred2 << "\n\n";
	}*/
	//.......................................................................................................
}