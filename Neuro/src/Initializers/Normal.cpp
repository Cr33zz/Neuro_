#include "Initializers/Normal.h"
#include "Tensors/Tensor.h"
#include "Tools.h"

namespace Neuro
{
	bool Normal::m_HasValue = false;
	float Normal::m_Value = 0;

	//////////////////////////////////////////////////////////////////////////
	Normal::Normal(float mean, float variance, float scale)
	{
		m_Mean = mean;
		m_Variance = variance;
		m_Scale = scale;
	}

	//////////////////////////////////////////////////////////////////////////
	float Normal::NextSingle(float mean, float stdDeviation, float scale)
	{
		//based upon https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/randomkit.c
		float variance = stdDeviation * stdDeviation;

		if (m_HasValue)
		{
			m_HasValue = false;
			return (variance * m_Value + mean) * scale;
		}

		float x1, x2, r2;
		do
		{
			x1 = 2 * GlobalRng().NextFloat() - 1;
			x2 = 2 * GlobalRng().NextFloat() - 1;
			r2 = x1 * x1 + x2 * x2;
		}
		while (r2 >= 1.0 || r2 == 0.0);

		//Polar method, a more efficient version of the Box-Muller approach.
		float f = (float)::sqrt(-2 * log(r2) / r2);

		m_HasValue = true;
		m_Value = f * x1;

		return (variance * (f * x2) + mean) * scale;
	}

    //////////////////////////////////////////////////////////////////////////
    float Normal::NextTruncatedSingle(float mean, float stdDeviation)
    {
        float x;
        do 
        {
            x = NextSingle(mean, stdDeviation);
        } while (abs(x - mean) > 2.f * stdDeviation);

        return x;
    }

    //////////////////////////////////////////////////////////////////////////
	void Normal::Init(Tensor& t)
	{
        t.FillWithFunc([&]() { return NextSingle(m_Mean, m_Variance, m_Scale); });
	}
}
