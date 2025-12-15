#pragma once
#include <vector>
#include <map>
#include <sstream>
#include <string>
#include <iostream>
#include <unordered_map>


class string_vector {
public:
	string_vector() = default;

	string_vector(const std::vector<std::string>& data) : vec_str(data) {}

	string_vector(size_t size, const std::string& value = std::string()) : vec_str(size, value) {}

	size_t size() const {
		return vec_str.size();
	}

	std::string& operator[](size_t index) {
		return vec_str[index];
	}

	const std::string& operator[](size_t index) const {
		return vec_str[index];
	}

	void push_back(const std::string& value) {
		vec_str.push_back(value);
	}

	const std::vector<std::string>& get_data() const {
		return vec_str;
	}

	std::vector<double> to_float() const {
		std::vector<double> result = as_float(this->vec_str);
		if (result.size() != this->vec_str.size()) {
			throw std::runtime_error("to_float conversion resulted in a vector of different size");
		}
		return result;
	}

	string_vector operator+(const string_vector& other) const {
		std::ostringstream stream;
		std::vector<double> vec1 = as_float(vec_str);
		std::vector<double> vec2 = as_float(other.vec_str);
		check_size(vec1, vec2);
		string_vector result(vec1.size());
		for (size_t i = 0; i < vec1.size(); ++i) {
			stream << vec1[i] + vec2[i];
			result[i] = stream.str();
			stream.str("");
		}
		return result;
	}

	string_vector operator-(const string_vector& other) const {
		std::ostringstream stream;
		std::vector<double> vec1 = as_float(vec_str);
		std::vector<double> vec2 = as_float(other.vec_str);
		check_size(vec1, vec2);
		string_vector result(vec1.size());
		for (size_t i = 0; i < vec1.size(); ++i) {
			stream << vec1[i] - vec2[i];
			result[i] = stream.str();
			stream.str("");
		}
		return result;
	}

	string_vector operator*(const string_vector& other) const {
		std::ostringstream stream;
		std::vector<double> vec1 = as_float(vec_str);
		std::vector<double> vec2 = as_float(other.vec_str);
		check_size(vec1, vec2);
		string_vector result(vec1.size());
		for (size_t i = 0; i < vec1.size(); ++i) {
			stream << vec1[i] * vec2[i];
			result[i] = stream.str();
			stream.str("");
		}
		return result;
	}

	string_vector operator/(const string_vector& other) const {
		std::ostringstream stream;
		std::vector<double> vec1 = as_float(vec_str);
		std::vector<double> vec2 = as_float(other.vec_str);
		check_size(vec1, vec2);
		string_vector result(vec1.size());
		for (size_t i = 0; i < vec1.size(); ++i) {
			if (vec2[i] == 0) {
				throw std::runtime_error("Division by zero in element-wise vector operation.");
			}
			stream << vec1[i] / (double)vec2[i];
			result[i] = stream.str();
			stream.str("");
		}
		return result;
	}

	// Addition (vector + scalar)
	string_vector operator+(const double& scalar) const {
		std::ostringstream stream;
		std::vector<double> vec1 = as_float(vec_str);
		string_vector result(vec1.size());
		for (size_t i = 0; i < vec1.size(); ++i) {
			stream << vec1[i] + scalar;
			result[i] = stream.str();
			stream.str("");
		}
		return result;
	}

	// Subtraction (vector - scalar)
	string_vector operator-(const double& scalar) const {
		std::ostringstream stream;
		std::vector<double> vec1 = as_float(vec_str);
		string_vector result(vec1.size());
		for (size_t i = 0; i < vec1.size(); ++i) {
			stream << vec1[i] - scalar;
			result[i] = stream.str();
			stream.str("");
		}
		return result;
	}

	// Multiplication (vector * scalar)
	string_vector operator*(const double& scalar) const {
		std::ostringstream stream;
		std::vector<double> vec1 = as_float(vec_str);
		string_vector result(vec1.size());
		for (size_t i = 0; i < vec1.size(); ++i) {
			stream << vec1[i] * scalar;
			result[i] = stream.str();
			stream.str("");
		}
		return result;
	}

	// Division (vector / scalar)
	string_vector operator/(const double& scalar) const {
		if (scalar == 0) {
			throw std::runtime_error("Division by zero in vector/scalar operation.");
		}
		std::ostringstream stream;
		std::vector<double> vec1 = as_float(vec_str);
		string_vector result(vec1.size());
		for (size_t i = 0; i < vec1.size(); ++i) {
			stream << vec1[i] / (double)scalar;
			result[i] = stream.str();
			stream.str("");
		}
		return result;
	}

	template <typename T>
	friend std::ostream& operator<<(std::ostream& os, const string_vector& v) {
		os << "[";
		for (size_t i = 0; i < v.size(); ++i) {
			os << v.vec_str[i];
			if (i < v.vec_str.size() - 1) {
				os << ", ";
			}
		}
		os << "]";
		return os;
	}

private:
	std::vector<std::string> vec_str;

	void check_size(const std::vector<double>& vec1, const std::vector<double>& vec2) const {
		if (vec1.size() != vec2.size()) {
			throw std::runtime_error("Vector sizes must match for element-wise operation.");
		}
	}

	std::vector<double> as_float(const std::vector<std::string>& vec) const {
		std::vector<double> result;
		result.reserve(vec.size());
		for (const auto& elem : vec) {
			double value = 0.0;
			try {
				value = elem.empty() ? -1.0 : std::stod(elem);
				result.push_back(value);
			}
			catch (const std::exception&) {
				continue;
			}
		}
		return result;
	}
};



using dataframe = std::unordered_map<std::string, string_vector>;

dataframe load_data(const std::string& filename) {
	auto split = [](const std::string& str, char delimiter) {
		std::vector<std::string> fields;
		std::stringstream ss(str);
		std::string field;

		while (std::getline(ss, field, delimiter)) {
			fields.push_back((field));
		}
		return fields;
		};

	std::fstream file{ filename, std::ios_base::in };
	dataframe spreadsheet;
	std::vector<std::string> header_names;

	std::string line;
	std::getline(file, line);

	for (std::string key : split(line, ',')) {
		header_names.push_back(key);
		spreadsheet[key] = string_vector{};
	}

	while (std::getline(file, line)) {
		std::vector<std::string> row = split(line, ',');
		for (size_t i{ 0 }; i < row.size(); ++i) {
			spreadsheet[header_names[i]].push_back(row[i]);
		}
	}
	return spreadsheet;
	file.close();
}


void save_to_csv(dataframe& data, const std::string& filename, const std::vector<std::string>& features) {
	if (data.empty()) {
		std::cerr << "Warning: Dataframe is empty. Nothing saved to file." << std::endl;
		return;
	}

	std::ofstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error("Could not open file for writing: " + filename);
	}

	std::vector<std::pair<std::string, const std::vector<std::string>*>> columns;
	size_t row_count = 0;
	bool first_column = true;

	for (const auto& key : features) {
		if (first_column) {
			row_count = data[key].size();
			first_column = false;
		}
		else if (data[key].size() != row_count) {
			throw std::runtime_error("Column '" + key + "' has a different size than the first column. All columns must be the same length.");
		}
		columns.push_back({ key, &data[key].get_data() });
	}

	// Write the Header Row
	for (size_t j = 0; j < columns.size(); ++j) {
		file << columns[j].first;
		if (j < columns.size() - 1) {
			file << ",";
		}
	}
	file << "\n";

	// Write Data Rows
	for (size_t i = 0; i < row_count; ++i) {
		for (size_t j = 0; j < columns.size(); ++j) {
			file << (*columns[j].second)[i];

			if (j < columns.size() - 1) {
				file << ",";
			}
		}
		file << "\n";
	}

	std::cout << "Successfully saved " << row_count << " rows to " << filename << std::endl;
}