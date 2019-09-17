#include "Initializers/Zeros.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	void Zeros::Init(Tensor& t)
	{
		t.Zero();
	}
}
