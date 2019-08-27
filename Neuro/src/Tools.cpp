#include <algorithm>
#include <iomanip>
#include <sstream>
#include <fstream>

#include "Tools.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	int AccNone(const Tensor& target, const Tensor& output)
	{
		return 0;
	}

	//////////////////////////////////////////////////////////////////////////
	int AccBinaryClassificationEquality(const Tensor& target, const Tensor& output)
	{
		int hits = 0;
		for (int n = 0; n < output.Batch(); ++n)
			hits += target(0, 0, 0, n) == roundf(output(0, 0, 0, n)) ? 1 : 0;
		return hits;
	}

	//////////////////////////////////////////////////////////////////////////
	int AccCategoricalClassificationEquality(const Tensor& target, const Tensor& output)
	{
        Tensor targetArgMax = target.ArgMax(EAxis::Sample);
        Tensor outputArgMax = output.ArgMax(EAxis::Sample);

		int hits = 0;
		for (int i = 0; i < targetArgMax.Length(); ++i)
			hits += targetArgMax(i) == outputArgMax(i) ? 1 : 0;
		return hits;
	}

    //////////////////////////////////////////////////////////////////////////
    void DeleteData(vector<tensor_ptr_vec_t>& data)
    {
        for (auto& v : data)
            DeleteContainer(v);
        data.clear();
    }

    //////////////////////////////////////////////////////////////////////////
	float Clip(float value, float min, float max)
	{
		return value < min ? min : (value > max ? max : value);
	}

	//////////////////////////////////////////////////////////////////////////
	int Sign(float value)
	{
		return value < 0 ? -1 : (value > 0 ? 1 : 0);
	}

	//////////////////////////////////////////////////////////////////////////
	std::string ToLower(const string& str)
	{
		string result = str;

		for (size_t i = 0; i < str.length(); ++i)
			result[i] = tolower(str[i]);

		return result;
	}

    //////////////////////////////////////////////////////////////////////////
    vector<string> Split(const string& str, const string& delimiter)
    {
        vector<string> result;
        size_t pos = 0;
        size_t lastPos = 0;

        while ((pos = str.find(delimiter, lastPos)) != string::npos)
        {
            result.push_back(str.substr(lastPos, pos - lastPos));
            lastPos = pos + delimiter.length();
        }

        result.push_back(str.substr(lastPos, str.length() - lastPos));

        return result;
    }

    //////////////////////////////////////////////////////////////////////////
	string GetProgressString(int iteration, int maxIterations, const string& extraStr, int barLength)
	{
		int maxIterLen = (int)to_string(maxIterations).length();
		float step = maxIterations / (float)barLength;
		int currStep = min((int)ceil(iteration / step), barLength);

		stringstream ss;
		ss << setw(maxIterLen) << iteration << "/" << maxIterations << " [";
		for (int i = 0; i < currStep - 1; ++i)
			ss << "=";
		ss << ((iteration == maxIterations) ? "=" : ">");
		for (int i = 0; i < barLength - currStep; ++i)
			ss << ".";
		ss << "] " << extraStr;
		return ss.str();
	}

    //////////////////////////////////////////////////////////////////////////
    void LoadCSVData(const string& filename, int outputsNum, Tensor& inputs, Tensor& outputs, bool outputsOneHotEncoded)
    {
        ifstream infile(filename.c_str());
        string line;

        vector<float> inputValues;
        vector<float> outputValues;
        int batches = 0;
        int inputBatchSize = 0;

        while (getline(infile, line))
        {
            if (line.empty())
                continue;

            auto tmp = Split(line, ",");

            ++batches;
            inputBatchSize = (int)tmp.size() - (outputsOneHotEncoded ? 1 : outputsNum);

            for (int i = 0; i < inputBatchSize; ++i)
                inputValues.push_back((float)atof(tmp[i].c_str()));

            for (int i = 0; i < (outputsOneHotEncoded ? 1 : outputsNum); ++i)
            {
                float v = (float)atof(tmp[inputBatchSize + i].c_str());

                if (outputsOneHotEncoded)
                {
                    for (int e = 0; e < outputsNum; ++e)
                        outputValues.push_back(e == (int)v ? 1.f : 0.f);
                }
                else
                    outputValues.push_back(v);
            }
        }

        inputs = Tensor(inputValues, Shape(1, inputBatchSize, 1, batches));
        outputs = Tensor(outputValues, Shape(1, outputsNum, 1, batches));
    }
}