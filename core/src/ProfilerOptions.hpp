#pragma once

#include "ProfilerOptions.hpp"
#include "getopt.h"
#include <iostream>
#include <random>
#include <fstream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pptree {
  enum OptionIds {
    OPT_SIM_MEAN = 256,
    OPT_SIM_MEAN_SEPARATION,
    OPT_SIM_SD
  };

  struct ProfilerOptions {
    int trees         = 100;
    float lambda      = 0.5;
    int threads       = -1;
    int seed          = -1;
    float p_vars      = 0.5;
    int n_vars        = -1;
    float train_ratio = 0.7;
    int n_runs        = 1;
    std::string data_path;
    std::string simulate;
    int rows                  = 1000;
    int cols                  = 10;
    int classes               = 2;
    float sim_mean            = 100.0f;
    float sim_mean_separation = 50.0f;
    float sim_sd              = 10.0f;
  };

  bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
  }

  void warn_unused_params(const ProfilerOptions& params) {
    if (params.trees == 0) {
      bool has_warnings = false;

      if (params.threads != -1) {
        std::cout << "Warning: threads parameter is ignored when training a single tree" << std::endl;
        has_warnings = true;
      }

      if (params.p_vars != 0.5) {
        std::cout << "Warning: var-proportion parameter is ignored when training a single tree" << std::endl;
        has_warnings = true;
      }

      if (has_warnings) {
        std::cout << "Single trees always use all features for splitting" << std::endl;
      }
    }
  }

  void init_params(ProfilerOptions& params, int total_vars = 0) {
    if (params.lambda == -1) {
      params.lambda = 0.5;
      std::cout << "Using default lambda: " << params.lambda << std::endl;
    }

    if (params.train_ratio <= 0 || params.train_ratio >= 1) {
      std::cerr << "Error: Train ratio must be between 0 and 1" << std::endl;
      exit(1);
    }

    if (params.seed == -1) {
      std::random_device rd;
      params.seed = rd();
      std::cout << "Using random seed: " << params.seed << std::endl;
    }

    if (params.threads == -1) {
     #ifdef _OPENMP
      params.threads = omp_get_max_threads();
     #else
      params.threads = 1;
     #endif
      std::cout << "Using default thread count: " << params.threads << std::endl;
    }

    if (total_vars > 0 && params.trees > 0) {
      if (params.p_vars == -1 && params.n_vars == -1) {
        params.p_vars = 0.5;
        params.n_vars = std::round(total_vars * params.p_vars);
        std::cout << "Using default variable proportion: " << params.p_vars  << " (" << params.n_vars << " variables)" << std::endl;
      } else if (params.p_vars != -1) {
        params.n_vars = std::round(total_vars * params.p_vars);
      } else {
        params.p_vars = static_cast<float>(params.n_vars) / total_vars;
      }
    }
  }

  void print_usage(const char *program) {
    std::cout << "Usage: " << program << " [Options]\n"
              << "\nOptions:\n"
              << "  -s, --simulate=NxMxK          Simulate NxM data matrix with K classes (instead of reading from file)\n"
              << "  -d, --data=PATH               CSV file to read (instead of simulating)\n"
              << "  -t, --trees=N                 Number of trees (default: 100, 0 for single tree)\n"
              << "  -l, --lambda=N                Method selection (0=LDA, (0,1]=PDA, default: 0.5)\n"
              << "  -n, --threads=N               Number of threads (default: CPU cores)\n"
              << "  -r, --seed=N                  Random seed (default: random)\n"
              << "  -v, --p-vars=F                Feature proportion for forest (default: 0.5)\n"
              << "  -m, --n-vars=N                Number of features to use per split\n"
              << "  -p, --train-ratio=F           Train set ratio (default: 0.7)\n"
              << "  -e, --n-runs=N                Number of training runs (default: 1)\n"
              << "\nSimulation parameters (only used with --simulate):\n"
              << "      --sim-mean=F              Mean for simulated data (default: 100.0)\n"
              << "      --sim-mean-separation=F   Mean separation between classes (default: 50.0)\n"
              << "      --sim-sd=F                Standard deviation for simulated data (default: 10.0)\n"
              << "  -h, --help                    Show this help message\n\n";
  }

  ProfilerOptions parse_args(int argc, char *argv[]) {
    ProfilerOptions params;
    const struct option long_options[] = {
      { "simulate",            required_argument,                0,                                's' },
      { "data",                required_argument,                0,                                'd' },
      { "trees",               required_argument,                0,                                't' },
      { "lambda",              required_argument,                0,                                'l' },
      { "threads",             required_argument,                0,                                'n' },
      { "seed",                required_argument,                0,                                'r' },
      { "p-vars",              required_argument,                0,                                'v' },
      { "n-vars",              required_argument,                0,                                'm' },
      { "train-ratio",         required_argument,                0,                                'p' },
      { "n-runs",              required_argument,                0,                                'e' },
      { "sim-mean",            required_argument,                0,                                OPT_SIM_MEAN },
      { "sim-mean-separation", required_argument,                0,                                OPT_SIM_MEAN_SEPARATION },
      { "sim-sd",              required_argument,                0,                                OPT_SIM_SD },
      { "help",                no_argument,                      0,                                'h' },
      { 0,                     0,                                0,                                0 }
    };

    bool has_simulate = false;
    bool has_data     = false;
    int option_index  = 0;
    int c;

    do {
      try {
        switch (c = getopt_long(argc, argv, "s:d:t:l:n:r:v:m:p:e:h", long_options, &option_index)) {
            case 's': {
              if (!optarg || strlen(optarg) == 0) {
                throw std::invalid_argument("--simulate requires a format NxMxK");
              }

              has_simulate = true;
              std::string sim_str = optarg;
              size_t x1           = sim_str.find('x');
              size_t x2           = sim_str.find('x', x1 + 1);

              if (x1 == std::string::npos || x2 == std::string::npos) {
                throw std::invalid_argument("Simulate format must be NxMxK (e.g., 1000x10x2)");
              }

              try {
                params.rows    = std::stoi(sim_str.substr(0, x1));
                params.cols    = std::stoi(sim_str.substr(x1 + 1, x2 - x1 - 1));
                params.classes = std::stoi(sim_str.substr(x2 + 1));

                if (params.rows <= 0 || params.cols <= 0 || params.classes <= 1) {
                  throw std::out_of_range("Values must be positive and classes must be > 1");
                }
              } catch (const std::exception& e) {
                throw std::invalid_argument(std::string("Invalid simulate values: ") + e.what());
              }
              break;
            }

            case 'd':

              if (!optarg || strlen(optarg) == 0) {
                throw std::invalid_argument("--data requires a valid file path");
              }

              if (!file_exists(optarg)) {
                throw std::invalid_argument(std::string("File not found: ") + optarg);
              }

              has_data         = true;
              params.data_path = optarg;
              break;

            case 't':

              if (!optarg || strlen(optarg) == 0) {
                throw std::invalid_argument("--trees requires a number");
              }

              try {
                params.trees = std::stoi(optarg);

                if (params.trees < 0) {
                  throw std::out_of_range("Number of trees must be non-negative");
                }
              } catch (const std::exception& e) {
                throw std::invalid_argument(std::string("Invalid trees value: ") + e.what());
              }
              break;

            case 'l':

              if (!optarg || strlen(optarg) == 0) {
                throw std::invalid_argument("--lambda requires a value between 0 and 1");
              }

              try {
                params.lambda = std::stof(optarg);

                if (params.lambda < 0 || params.lambda > 1) {
                  throw std::out_of_range("Lambda must be between 0 and 1");
                }
              } catch (const std::exception& e) {
                throw std::invalid_argument(std::string("Invalid lambda value: ") + e.what());
              }
              break;

            case 'n':

              if (!optarg || strlen(optarg) == 0) {
                throw std::invalid_argument("--threads requires a positive number");
              }

              try {
                params.threads = std::stoi(optarg);

                if (params.threads < 1) {
                  throw std::out_of_range("Number of threads must be positive");
                }
              } catch (const std::exception& e) {
                throw std::invalid_argument(std::string("Invalid threads value: ") + e.what());
              }
              break;

            case 'r':
              params.seed = std::stoi(optarg);
              break;

            case 'v':

              if (!optarg || strlen(optarg) == 0) {
                throw std::invalid_argument("--p-vars requires a value between 0 and 1");
              }

              try {
                params.p_vars = std::stof(optarg);

                if (params.p_vars <= 0 || params.p_vars > 1) {
                  throw std::out_of_range("Variable proportion must be between 0 and 1");
                }
              } catch (const std::exception& e) {
                throw std::invalid_argument(std::string("Invalid variable proportion: ") + e.what());
              }
              break;

            case 'm':
              params.n_vars = std::stoi(optarg);
              break;

            case 'p':

              if (!optarg || strlen(optarg) == 0) {
                throw std::invalid_argument("--train-ratio requires a value between 0 and 1");
              }

              try {
                params.train_ratio = std::stof(optarg);

                if (params.train_ratio <= 0 || params.train_ratio >= 1) {
                  throw std::out_of_range("Train ratio must be between 0 and 1");
                }
              } catch (const std::exception& e) {
                throw std::invalid_argument(std::string("Invalid train ratio: ") + e.what());
              }
              break;

            case 'e':

              if (!optarg || strlen(optarg) == 0) {
                throw std::invalid_argument("--n-runs requires a positive number");
              }

              try {
                params.n_runs = std::stoi(optarg);

                if (params.n_runs < 1) {
                  throw std::out_of_range("Number of runs must be positive");
                }
              } catch (const std::exception& e) {
                throw std::invalid_argument(std::string("Invalid number of runs: ") + e.what());
              }
              break;

            case OPT_SIM_MEAN_SEPARATION:

              if (!optarg || strlen(optarg) == 0) {
                throw std::invalid_argument("--sim-mean-separation requires a numeric value");
              }

              try {
                params.sim_mean_separation = std::stof(optarg);

                if (params.sim_mean_separation <= 0) {
                  throw std::out_of_range("Simulation mean separation must be positive");
                }
              } catch (const std::exception& e) {
                throw std::invalid_argument(std::string("Invalid simulation mean separation: ") + e.what());
              }
              break;

            case 'h':
              print_usage(argv[0]);
              exit(0);
              break;

            case -1:
              break;

            default:
              std::cerr << "Error: Invalid option " << c << std::endl;
              exit(1);
        }
      } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "Use --help for usage information" << std::endl;
        exit(1);
      }
    } while (c != -1);

    if (!has_simulate && !has_data) {
      std::cerr << "Error: Must specify either --simulate or --data" << std::endl;
      std::cerr << "Use --help for usage information" << std::endl;
      exit(1);
    }

    if (has_simulate && has_data) {
      std::cerr << "Error: Cannot specify both --simulate and --data" << std::endl;
      std::cerr << "Use --help for usage information" << std::endl;
      exit(1);
    }

    warn_unused_params(params);
    return params;
  }
}
