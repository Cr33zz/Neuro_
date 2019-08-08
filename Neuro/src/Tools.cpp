#include <algorithm>
#include <iomanip>
#include <sstream>

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
		for (int n = 0; n < output.BatchSize(); ++n)
			hits += target(0, 0, 0, n) == roundf(output(0, 0, 0, n)) ? 1 : 0;
		return hits;
	}

	//////////////////////////////////////////////////////////////////////////
	int AccCategoricalClassificationEquality(const Tensor& target, const Tensor& output)
	{
		int hits = 0;
		for (int n = 0; n < output.BatchSize(); ++n)
			hits += target.ArgMax(n) == output.ArgMax(n) ? 1 : 0;
		return hits;
	}

	//////////////////////////////////////////////////////////////////////////
	void Delete(tensor_ptr_vec_t& tensorsVec)
	{
		for (auto tPtr : tensorsVec)
			delete tPtr;
		tensorsVec.clear();
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
}