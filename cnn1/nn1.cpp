#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <string>

using namespace torch;
using namespace std;

struct AlexNetImpl : torch::nn::Module {
  // MODULE LAYERS
  torch::nn::Linear linear1, linear2;
  AlexNetImpl(int64_t N) : 
    linear1(register_module("linear1", nn::Linear(N, 64))),
    linear2(register_module("linear2", nn::Linear(64, 1)))
    {}
          
  torch::Tensor forward(const torch::Tensor& input) {
    auto x = torch::relu(linear1(input));
    x = linear2(x);
    return x;
  }
};

TORCH_MODULE_IMPL(AlexNet, AlexNetImpl);

// parse csv file
vector<vector<double>> read_csv(string loc){
  vector<vector<double>> ans;

  ifstream data(loc);
  if (!data.is_open()){
    exit(EXIT_FAILURE);
  }
  string str;
  getline(data, str); // skip the first line
  while (getline(data, str)){
    vector<double> line;
    istringstream iss(str);
    string token;
    while (getline(iss, token, ',')){
      std::size_t found = token.find(')');
      if( found != std::string::npos)
        token = token.substr(found+1);
      line.push_back(atof(token.c_str()));
    }
    ans.push_back(line);
  }
  return ans;
}

//sample data for a banchsize 
std::pair<torch::Tensor, torch::Tensor> sample(vector<vector<double>>& data, int batch_size, int input_length){
  torch::Tensor x = torch::ones({batch_size, input_length});
  torch::Tensor target = torch::ones({batch_size, 1});
  for(int batch = 0; batch < batch_size; ++batch){
    int sample = rand() % data.size();
    x[batch] = torch::from_blob(data[sample].data(), {input_length}, torch::kF64).clone();
    target[batch] = torch::from_blob(data[sample].data()+input_length, {1}, torch::kF64).clone();
  }
  return make_pair(x, target);
}


int main(int argc, char **argv) {

  torch::Device device = torch::kCPU;

  #ifdef DEBUG
    std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
  #endif

  if (torch::cuda::is_available()) {
    #ifdef DEBUG
      std::cout << "CUDA is available! Training on GPU." << std::endl;
    #endif
    device = torch::kCUDA;
  }

  vector<vector<double>> trian_data = read_csv("trian.csv");
  vector<vector<double>> test_data = read_csv("test.csv");

  int64_t input_length = trian_data[0].size()-1;
  int batch_size = stoi(argv[1]);
  int eposides = stoi(argv[2]);
  auto model = AlexNet(input_length);
  torch::optim::Adam optim(model->parameters(), torch::optim::AdamOptions(1e-3));

  model->train();
  model->to(device);
  std::ofstream perf; 
  perf.open("tiger.csv");
  perf << "eposides,"<<"loss"<<std::endl;
  for (int i = 0; i < eposides; ++i) {
    //trian
    auto samp = sample(trian_data, batch_size, input_length);
    optim.zero_grad();
    torch::Tensor y = model->forward(samp.first);
    torch::Tensor loss = torch::mse_loss(y, samp.second);
    loss.backward();
    optim.step();
    //test
    if(i%100 == 0){
      auto samp = sample(test_data, batch_size, input_length);
      torch::Tensor loss_evl = torch::mse_loss(model->forward(samp.first), samp.second);
      #ifdef DEBUG
        cout << "\033[1m\033[35m\rEposide #" << i << "\tLoss" << *static_cast<float*>(loss_evl.storage().data()) << "\033[0m" << endl;
      #else
        perf << i << "," << *static_cast<float*>(loss_evl.storage().data()) << std::endl;
      #endif
    }
    // if(i%10 == 0) cout << "eposide 0"*static_cast<float*>(loss.storage().data()) << flush;
  }
  for(int i = 0; i < 100; i++){
    auto samp = sample(test_data, batch_size, input_length);
    torch::Tensor loss_evl = torch::mse_loss(model->forward(samp.first), samp.second);
    perf << i*100+eposides << "," << *static_cast<float*>(loss_evl.storage().data()) << std::endl;
  }
  perf.close();
}

