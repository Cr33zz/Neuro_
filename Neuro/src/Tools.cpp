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
		int hits = 0;
		for (int n = 0; n < output.Batch(); ++n)
			hits += target.ArgMax(n) == output.ArgMax(n) ? 1 : 0;
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
            result.push_back(str.substr(lastPos, pos));
            lastPos = pos + delimiter.length();
        }

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
    void LoadCSVData(const string& filename, int outputsNum, vector<tensor_ptr_vec_t>& inputs, vector<tensor_ptr_vec_t>& outputs, bool outputsOneHotEncoded /*= false*/)
    {
        ifstream infile(filename.c_str());
        string line;

        while (getline(infile, line))
        {
            auto tmp = Split(line, ",");

            auto input = new Tensor(Shape(1, (int)tmp.size() - (outputsOneHotEncoded ? 1 : outputsNum)));
            auto output = new Tensor(Shape(1, outputsNum));

            for (int i = 0; i < input->Length(); ++i)
                (*input)(0, i) = (float)atof(tmp[i].c_str());

            for (int i = 0; i < (outputsOneHotEncoded ? 1 : outputsNum); ++i)
            {
                float v = (float)atof(tmp[input->Length() + i].c_str());
                if (outputsOneHotEncoded)
                    (*output)(0, (int)v) = 1;
                else
                    (*output)(0, i) = v;
            }

            inputs.push_back({ input });
            outputs.push_back({ output });
        }
    }
}