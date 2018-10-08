#include "htmhelper.hpp"

#include <fstream>

#include <xtensor/xcsv.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>

const int LEN_INPUT_SDR = 512;
const int LEN_ENCODE = 24;
const int TP_DEPTH = 32;
const int NUM_BIT_DIFF = 8;

xt::xarray<bool> encode(float value, int encode_length, int sdr_lendth)
{
	xt::xarray<bool> base = xt::zeros<bool>({sdr_lendth});
	float percrent = sdr_lendth - encode_length;
	int start = percrent*value;
	int end = start + encode_length;
	auto v = xt::view(base, xt::range(start, end));
	v = true;
	return base;
}


struct Model
{
	Model(): tm({LEN_INPUT_SDR}, TP_DEPTH, 255, 255)
	{
		//tm->setMinThreshold(LEN_INPUT_SDR*0.35f+1);
		//tm->setActivationThreshold(LEN_INPUT_SDR*0.75f);
		//tm->setMaxNewSynapseCount(1024);
		tm->setPermanenceIncrement(0.95);
		tm->setPermanenceDecrement(0.082);
		tm->setConnectedPermanence(0.23);
		tm->setPredictedSegmentDecrement((1.f/LEN_INPUT_SDR)*tm->getPermanenceIncrement()*2.5+0.001);
		tm->setCheckInputs(false);
	}

	xt::xarray<bool> train(const xt::xarray<bool>& x) {return compute(x, true);}
	xt::xarray<bool> predict(const xt::xarray<bool>& x) {return compute(x, false);}
	xt::xarray<bool> compute(const xt::xarray<bool>& x, bool learn) {return tm.compute(x, learn);}
	void reset() {tm.reset();}

	TM tm;
};

xt::xarray<float> loadCSV(std::string path)
{
	std::ifstream in(path);
	xt::xarray<float> dataset = xt::load_csv<float>(in);
	return dataset;
}

int testModel(Model& model, const xt::xarray<float>& dataset)
{
	int problem_dataset = 0;
	Anom anomaly;
	for(size_t i=0;i<dataset.shape()[0];i++) {
		int num_anomaly = 0;
		float amno_sum = 0;
		for(size_t j=0;j<dataset.shape()[1]-1;j++) {
			float val = xt::view(dataset, i, j)[0];
			float next_val = xt::view(dataset, i, j+1)[0];
			xt::xarray<bool> in = encode(val, LEN_ENCODE, LEN_INPUT_SDR);
			xt::xarray<bool> prediction = model.predict(in);
			xt::xarray<bool> y = encode(next_val, LEN_ENCODE, LEN_INPUT_SDR);
			float amno = anomaly(y, prediction);
			float max_amno = NUM_BIT_DIFF/(float)LEN_ENCODE;
			if(amno >= max_amno)
				num_anomaly += 1;
			amno_sum += amno;
		}
	
		model.reset();
		if(amno_sum > 14) {
			problem_dataset += 1;
			std::cout << "Test " << i << ", anomaly detected: " << num_anomaly << ", err sum - " << amno_sum << std::endl;
		}
	}
	return problem_dataset;
}

void trainDataset(Model& model, const xt::xarray<float>& data_seq)
{
	for(size_t i=0;i<data_seq.shape()[0];i++) {
		for(size_t j=0;j<data_seq.shape()[1];j++) {
			float val = xt::view(data_seq, i, j)[0];
			model.train(encode(val, LEN_ENCODE, LEN_INPUT_SDR));
		}
		
		model.reset();
		std::cout << "\rTrain " << i << std::flush;
	}
}

int main()
{
	xt::xarray<float> dataset = loadCSV("ptbdb_normal.csv");
	
	xt::xarray<float> data_seq = xt::view(dataset, xt::range(0, 900), xt::range(0, -1));
	std::cout << "# input sequences = " << data_seq.shape()[0] << std::endl;
	std::cout << "# input features = " << data_seq.shape()[1] << std::endl;
	
	Model model;
	
	//for(int i=0;i<3;i++)
	trainDataset(model, data_seq);
	
	std::cout << "\n";
	
	xt::xarray<float> test_dataset = loadCSV("ptbdb_abnormal.csv");
	xt::xarray<float> test_seq = xt::view(test_dataset, xt::range(0, -1));
	
	data_seq = xt::view(dataset, xt::range(1000,1100),  xt::range(0, -1));
	test_seq = xt::view(test_seq, xt::range(0,100));
	
	float train_success_rate = testModel(model, data_seq)/(float)data_seq.shape()[0];
	std::cout << std::endl;
	float test_success_rate = testModel(model, test_seq)/(float)test_seq.shape()[0];
	
	std::cout << "Positive set anmoaly: " << train_success_rate << std::endl;
	std::cout << "Negative set anmoaly: " << test_success_rate << std::endl;
}
