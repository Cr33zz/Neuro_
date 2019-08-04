#include "Initializers/Zeros.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	void Zeros::Init(Tensor& t, int fanIn, int fanOut)
	{
		t.Zero();
	}
}
