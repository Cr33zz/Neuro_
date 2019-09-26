#include "Layers/Flatten.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Flatten::Flatten(LayerBase* inputLayer, const string& name)
        : Reshape(__FUNCTION__, inputLayer, Shape(inputLayer->OutputShape().Length), name)
	{
	}

    //////////////////////////////////////////////////////////////////////////
    Flatten::Flatten(const string& name)
        : Reshape(__FUNCTION__, Shape(), name)
    {
    }

	//////////////////////////////////////////////////////////////////////////
	Flatten::Flatten(const Shape& inputShape, const string& name)
		: Reshape(__FUNCTION__, inputShape, Shape(inputShape.Length), name)
	{
	}

    //////////////////////////////////////////////////////////////////////////
	LayerBase* Flatten::GetCloneInstance() const
	{
		return new Flatten(false);
	}

    //////////////////////////////////////////////////////////////////////////
    void Flatten::OnLinkInput(const vector<LayerBase*>& inputLayers)
    {
        __super::OnLinkInput(inputLayers);

        m_OutputsShapes[0] = Shape(m_InputShape.Length);
    }
}
