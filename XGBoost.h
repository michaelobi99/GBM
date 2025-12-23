#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

struct Node {
	bool leaf = false;
	int feature = -1;
};