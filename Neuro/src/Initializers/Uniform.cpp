﻿#include "Initializers/Uniform.h"
#include "Tensors/Tensor.h"
#include "Tools.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Uniform::Uniform(float min, float max)
	{
		m_Min = min;
		m_Max = max;
	}

	//////////////////////////////////////////////////////////////////////////
	float Uniform::NextSingle(float min, float max)
	{
		return min + GlobalRng().NextFloat() * (max - min);
	}

	//////////////////////////////////////////////////////////////////////////
	void Uniform::Init(Tensor& t)
	{
        t.FillWithFunc([&]() { return NextSingle(m_Min, m_Max); });
	}
}
