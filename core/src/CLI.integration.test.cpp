#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <unistd.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

#ifndef PPTREE_BINARY_PATH
#error "PPTREE_BINARY_PATH must be defined"
#endif

#ifndef PPTREE_DATA_DIR
#error "PPTREE_DATA_DIR must be defined"
#endif

static const std::string BINARY   = PPTREE_BINARY_PATH;
static const std::string DATA_DIR = PPTREE_DATA_DIR;
static const std::string IRIS_CSV = DATA_DIR + "/iris.csv";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct ProcessResult {
  int exit_code;
  std::string stdout_output;
};

static ProcessResult run_pptree(const std::string& args) {
  std::string cmd = BINARY + " " + args + " 2>/dev/null";

  FILE* pipe = popen(cmd.c_str(), "r");
  if (!pipe) {
    return {-1, ""};
  }

  std::string output;
  char buffer[4096];

  while (fgets(buffer, sizeof(buffer), pipe)) {
    output += buffer;
  }

  int status = pclose(pipe);
  int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;

  return {exit_code, output};
}

class TempFile {
  public:
    TempFile(const std::string& suffix = ".json") {
      std::string tmpl = "/tmp/pptree_test_XXXXXX" + suffix;
      std::vector<char> tmpl_buf(tmpl.begin(), tmpl.end());
      tmpl_buf.push_back('\0');

      int fd = mkstemps(tmpl_buf.data(), static_cast<int>(suffix.size()));

      if (fd != -1) {
        path_ = tmpl_buf.data();
        close(fd);
      }
    }

    ~TempFile() {
      if (!path_.empty()) {
        std::remove(path_.c_str());
      }
    }

    const std::string& path() const { return path_; }

    std::string read() const {
      std::ifstream in(path_);
      std::stringstream ss;
      ss << in.rdbuf();
      return ss.str();
    }

  private:
    std::string path_;
};

// ---------------------------------------------------------------------------
// Train
// ---------------------------------------------------------------------------

TEST(CLIIntegration, TrainWithIrisData) {
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(CLIIntegration, TrainAndSaveForest) {
  TempFile model;
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -o " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["model_type"], "forest");
  EXPECT_EQ(j["trees"], 5);
  EXPECT_TRUE(j.contains("lambda"));
  EXPECT_TRUE(j.contains("model"));
}

TEST(CLIIntegration, TrainAndSaveSingleTree) {
  TempFile model;
  auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 0 -o " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["model_type"], "tree");
}

TEST(CLIIntegration, TrainJsonOutput) {
  TempFile model;
  auto result = run_pptree("--output-format=json train -d " + IRIS_CSV + " -t 5 -r 42 -o " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(result.stdout_output);
  EXPECT_EQ(j["model_type"], "forest");
  EXPECT_EQ(j["saved"], true);
}

TEST(CLIIntegration, TrainMissingDataFails) {
  auto result = run_pptree("train");
  EXPECT_NE(result.exit_code, 0);
}

TEST(CLIIntegration, TrainNonexistentFileFails) {
  auto result = run_pptree("train -d /nonexistent/file.csv");
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Predict
// ---------------------------------------------------------------------------

class PredictTest : public ::testing::Test {
  protected:
    void SetUp() override {
      model_.reset(new TempFile());
      auto result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -o " + model_->path());
      ASSERT_EQ(result.exit_code, 0);
    }

    std::unique_ptr<TempFile> model_;
};

TEST_F(PredictTest, PredictWithSavedModel) {
  auto result = run_pptree("-q predict -M " + model_->path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_FALSE(result.stdout_output.empty());
  EXPECT_NE(result.stdout_output.find("Error rate:"), std::string::npos);
}

TEST_F(PredictTest, PredictJsonOutput) {
  auto result = run_pptree("--output-format=json predict -M " + model_->path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(result.stdout_output);
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_TRUE(j["predictions"].is_array());
  EXPECT_TRUE(j.contains("error_rate"));
}

TEST(CLIIntegration, PredictMissingModelArgFails) {
  auto result = run_pptree("predict -d " + IRIS_CSV);
  EXPECT_NE(result.exit_code, 0);
}

TEST(CLIIntegration, PredictNonexistentModelFails) {
  auto result = run_pptree("predict -M /nonexistent.json -d " + IRIS_CSV);
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Evaluate
// ---------------------------------------------------------------------------

TEST(CLIIntegration, EvaluateWithSimulatedData) {
  auto result = run_pptree("-q evaluate -s 50x3x2 -t 5 -r 42");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(CLIIntegration, EvaluateWithIrisData) {
  auto result = run_pptree("-q evaluate -d " + IRIS_CSV + " -t 5 -r 42");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(CLIIntegration, EvaluateTextOutput) {
  auto result = run_pptree("-q evaluate -s 50x3x2 -t 5 -r 42");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("Evaluation results"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Train Error"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Test Error"), std::string::npos);
}

TEST(CLIIntegration, EvaluateJsonOutput) {
  auto result = run_pptree("--output-format=json evaluate -s 50x3x2 -t 5 -r 42");
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(result.stdout_output);
  EXPECT_TRUE(j.contains("runs"));
  EXPECT_TRUE(j.contains("mean_time_ms"));
  EXPECT_TRUE(j.contains("mean_train_error"));
  EXPECT_TRUE(j.contains("mean_test_error"));
}

TEST(CLIIntegration, EvaluateMultipleRuns) {
  auto result = run_pptree("--output-format=json evaluate -s 50x3x2 -t 5 -e 3 -r 42");
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(result.stdout_output);
  EXPECT_EQ(j["runs"], 3);
}

TEST(CLIIntegration, EvaluateSingleTree) {
  auto result = run_pptree("-q evaluate -s 50x3x2 -t 0 -r 42");
  EXPECT_EQ(result.exit_code, 0);
}

TEST(CLIIntegration, EvaluateNoDataSourceFails) {
  auto result = run_pptree("evaluate");
  EXPECT_NE(result.exit_code, 0);
}

TEST(CLIIntegration, EvaluateInvalidSimFormatFails) {
  auto result = run_pptree("evaluate -s 100x5");
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Global Options
// ---------------------------------------------------------------------------

TEST(CLIIntegration, HelpFlag) {
  auto result = run_pptree("--help");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("train"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("predict"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("evaluate"), std::string::npos);
}

TEST(CLIIntegration, QuietSuppressesOutput) {
  auto quiet_result = run_pptree("-q evaluate -s 50x3x2 -t 5 -r 42");
  EXPECT_EQ(quiet_result.exit_code, 0);
  EXPECT_EQ(quiet_result.stdout_output.find("Training random forest"), std::string::npos);
  EXPECT_EQ(quiet_result.stdout_output.find("Using default"), std::string::npos);
}

TEST(CLIIntegration, NoArgsExitsNonZero) {
  auto result = run_pptree("");
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Config File
// ---------------------------------------------------------------------------

TEST(CLIIntegration, ConfigFileApplied) {
  TempFile config;
  {
    std::ofstream out(config.path());
    out << R"({"trees": 3})";
  }

  auto result = run_pptree("--config " + config.path() + " --output-format=json evaluate -s 50x3x2 -r 42");
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(result.stdout_output);
  EXPECT_TRUE(j.contains("runs"));
}

// ---------------------------------------------------------------------------
// End-to-End Pipeline
// ---------------------------------------------------------------------------

TEST(CLIIntegration, TrainThenPredict) {
  TempFile model;

  auto train_result = run_pptree("-q train -d " + IRIS_CSV + " -t 5 -r 42 -o " + model.path());
  ASSERT_EQ(train_result.exit_code, 0);

  auto predict_result = run_pptree("--output-format=json predict -M " + model.path() + " -d " + IRIS_CSV);
  ASSERT_EQ(predict_result.exit_code, 0);

  auto j = json::parse(predict_result.stdout_output);
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0u);
  EXPECT_TRUE(j.contains("error_rate"));

  for (const auto& pred : j["predictions"]) {
    EXPECT_TRUE(pred.is_number_integer());
  }
}
