#include "pptree.hpp"

using namespace pptree;

DataSpec<long double, int> simulate(
  const int n,
  const int p,
  const int G) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<long double> norm(0, 1);

  Data<long double> x(n, p);

  for (int i = 0; i < x.rows(); ++i) {
    for (int j = 0; j < x.cols(); ++j) {
      x(i, j) = norm(gen);
    }
  }

  DataColumn<int> y(n);

  for (int i = 0; i < y.size(); ++i) {
    y[i] = i % G;
  }

  return DataSpec<long double, int>(x, y);
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " n p G B" << std::endl;
    return 1;
  }

  const int n = std::stoi(argv[1]);
  const int p = std::stoi(argv[2]);
  const int G = std::stoi(argv[3]);
  const int B = std::stoi(argv[4]);

  const auto data = simulate(n, p, G);
  const auto spec = TrainingSpec<long double, int>::uniform_glda(std::round((std::sqrt(p - 1) / (p - 1)) * p), 0.1);

  if (B > 1) {
    Forest<long double, int>::train(*spec, data, B, 0);
  } else {
    Tree<long double, int>::train(*spec, data);
  }

  return 0;
}
