#include "Initializers/Constant.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Constant::Constant(float value /*= 1*/)
	{
		Value = value;
	}

	//////////////////////////////////////////////////////////////////////////
	void Constant::Init(Tensor& t, int fanIn, int fanOut)
	{
        t.FillWithValue(Value);
	}
}
