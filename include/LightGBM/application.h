#ifndef LIGHTGBM_APPLICATION_H_
#define LIGHTGBM_APPLICATION_H_

#include <LightGBM/meta.h>
#include <LightGBM/config.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/text_reader.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <LightGBM/network.h>
#include <LightGBM/dataset.h>
#include <LightGBM/dataset_loader.h>
#include <LightGBM/boosting.h>
#include <LightGBM/objective_function.h>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/metric.h>
#include <LightGBM/predictor.hpp>

#include <vector>
#include <memory>
#include <string>
#include <iostream>

namespace LightGBM {
/*!
* \brief The main entrance of LightGBM. this application has two tasks:
*        Train and Predict.
*        Train task will train a new model
*        Predict task will predict the scores of test data using existing model,
*        and save the score to disk.
*/
class Application {
public:
  Application(int argc, char** argv);

  /*! \brief Destructor */
  ~Application();

  /*! \brief To call this funciton to run application*/
  inline void Run();
  
  /*! \brief Initializations before prediction */
  void InitPredict();
  
  std::string Predict(const char* sample, const char sep);

  std::string Predict(const std::vector<std::pair<int, double>>& indexed_features);

  std::string Predict(const std::vector<double>& features);

  std::vector<std::string> BatchPredict(const std::vector<std::vector<double>>& batch_features);

private:

  /*! \brief Load parameters from command line and config file*/
  void LoadParameters(int argc, char** argv);

  /*! \brief Load data, including training data and validation data*/
  void LoadData();

  /*! \brief Initialization before training*/
  void InitTrain();

  /*! \brief Main Training logic */
  void Train();

  /*! \brief Main predicting logic */
  void Predict();

  /*! \brief Main Convert model logic */
  void ConvertModel();

  /*! \brief All configs */
  Config config_;
  /*! \brief Training data */
  std::unique_ptr<Dataset> train_data_;
  /*! \brief Validation data */
  std::vector<std::unique_ptr<Dataset>> valid_datas_;
  /*! \brief Metric for training data */
  std::vector<std::unique_ptr<Metric>> train_metric_;
  /*! \brief Metrics for validation data */
  std::vector<std::vector<std::unique_ptr<Metric>>> valid_metrics_;
  /*! \brief Boosting object */
  std::unique_ptr<Boosting> boosting_;
  /*! \brief Training objective function */
  std::unique_ptr<ObjectiveFunction> objective_fun_;
  /*! \brief Trained predictor */
  std::unique_ptr<Predictor> predictor_;
};


inline void Application::Run() {
  if (config_.task == TaskType::kPredict || config_.task == TaskType::KRefitTree) {
    InitPredict();
    Predict();
  } else if (config_.task == TaskType::kConvertModel) {
    ConvertModel();
  } else {
    InitTrain();
    Train();
  }
}

}  // namespace LightGBM

#endif   // LightGBM_APPLICATION_H_
