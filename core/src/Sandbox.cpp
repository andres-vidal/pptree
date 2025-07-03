#include "pptree.hpp"

#include "DataPacket.hpp"

#include "csv.hpp"
#include <vector>
#include <random>
#include <unordered_map>
#include <iostream>
#include <numeric>
#include <cmath>

#define T double
#define R int

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace pptree;
using namespace models::stats;
using namespace models;

DataPacket<T, R> read_csv(const std::string& filename) {
  csv::CSVReader reader(filename);
  std::vector<std::vector<T> > featureData;
  std::vector<std::string> rawLabels;

  for (csv::CSVRow& row : reader) {
    if (row.size() < 1) {
      throw std::runtime_error("CSV row has no columns.");
    }

    std::vector<T> currentFeatures;
    for (int j = 0; j < row.size() - 1; ++j) {
      currentFeatures.push_back(row[j].get<T>());
    }

    featureData.push_back(std::move(currentFeatures));

    // Read the last column as a string.
    std::string labelStr = row[row.size() - 1].get<std::string>();
    rawLabels.push_back(labelStr);
  }

  if (featureData.empty()) {
    throw std::runtime_error("CSV file is empty.");
  }

  // Map string labels to integer codes.
  std::unordered_map<std::string, int> labelMapping;
  std::vector<int> labels;
  int labelIndex = 0;
  for (const auto &labelStr : rawLabels) {
    if (labelMapping.find(labelStr) == labelMapping.end()) {
      labelMapping[labelStr] = labelIndex++;
    }

    labels.push_back(labelMapping[labelStr]);
  }

  // Determine dimensions for feature matrix.
  const int n = featureData.size();
  const int p = featureData[0].size();

  Data<T> x(n, p);
  for (int i = 0; i < n; ++i) {
    if (featureData[i].size() != p) {
      throw std::runtime_error("Inconsistent number of feature columns in CSV file.");
    }

    for (int j = 0; j < p; ++j) {
      x(i, j) = featureData[i][j];
    }
  }

  DataColumn<int> y(n);
  for (int i = 0; i < n; ++i) {
    y[i] = labels[i];
  }

  return DataPacket<T, R>(x, y);
}

void display_progress(int current, int total, int bar_width = 50) {
  float progress = static_cast<float>(current) / total;
  int pos        = static_cast<int>(bar_width * progress);

  std::cout << "\r" << std::string(pos, '-') << std::string(bar_width - pos, ' ')
            << " | " << current << "/" << total
            << " (" << static_cast<int>(progress * 100.0) << "%)     " << std::flush;

  if (current == total) {
    std::cout << std::endl;
  }
}

int main(int argc, char *argv[]) {
  const int reps = 10000;
  const int size = 100;

  DataPacket<T, R> data = read_csv("data/NCI60.csv");

  Data<T>  x       = data.x;
  DataColumn<R>  y = data.y;

  GroupSpec<R> group_spec(y);

  TrainingSpecUGLDA<T, R> training_spec(30, 0.5);


  #ifdef _OPENMP
  omp_set_num_threads(24);
  #endif

  invariant(size > 0, "The forest size must be greater than 0.");

  std::vector<BootstrapTreePtr<T, R> > trees(size);

  const int seed = 42;

  std::cout << "Running " << reps << " iterations:" << std::endl;

  for (int i = 0; i < reps; i++) {
    display_progress(i + 1, reps);


    #pragma omp parallel for schedule(dynamic) firstprivate(x, y)
    for (int i = 0; i < size; i++) {
      stats::RNG rng(static_cast<uint64_t>(seed), static_cast<uint64_t>(i));

      Tree<T, R>::train(training_spec, x, y, rng);
    }
  }
}
