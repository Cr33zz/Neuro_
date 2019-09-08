#include "Layers/Flatten.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Flatten::Flatten(LayerBase* inputLayer, const string& name)
        : Reshape(__FUNCTION__, inputLayer, Shape(1, inputLayer->OutputShape().Length), name)
	{
	}

    //////////////////////////////////////////////////////////////////////////
    Flatten::Flatten(const string& name)
        : Reshape(__FUNCTION__, Shape(), name)
    {
    }

	//////////////////////////////////////////////////////////////////////////
	Flatten::Flatten(const Shape& inputShape, const string& name)
		: Reshape(__FUNCTION__, inputShape, Shape(1, inputShape.Length), name)
	{
	}

    //////////////////////////////////////////////////////////////////////////
	LayerBase* Flatten::GetCloneInstance() const
	{
		return new Flatten(false);
	}

    //////////////////////////////////////////////////////////////////////////
    void Flatten::OnLink()
    {
        m_OutputShapes[0] = Shape(1, m_InputShapes[0].Length);
    }
}
