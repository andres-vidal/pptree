#include "pptree.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif


using namespace pptree;

DataSpec<double, int> simulate(
  const int n,
  const int p,
  const int G) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> norm(0, 1);

  Data<double> x(n, p);

  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.cols(); ++j) {
      x(i, j) = norm(gen);
    }
  }

  DataColumn<int> y(n);

  for (int i = 0; i < y.size(); ++i) {
    y[i] = i % G;
  }

  return DataSpec<double, int>(x, y);
}

int main(int argc, char *argv[]) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0] << " n p G B C" << std::endl;
    return 1;
  }

  const int n = std::stoi(argv[1]);
  const int p = std::stoi(argv[2]);
  const int G = std::stoi(argv[3]);
  const int B = std::stoi(argv[4]);
  const int C = std::stoi(argv[5]);

  const auto data = simulate(n, p, G);
  const auto spec = TrainingSpec<double, int>::uniform_glda(std::round(p / 2), 0.1);

  const auto start = std::chrono::high_resolution_clock::now();

  if (B > 1) {
    Forest<double, int>::train(*spec, data, B, 0, C);
  } else {
    Tree<double, int>::train(*spec, data);
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

  std::cout << "Elapsed Time: " << elapsed_time << " seconds." << std::endl;

  return 0;
}
