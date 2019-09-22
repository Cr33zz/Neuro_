#include <sstream>

#include "Activations.h"
#include "Layers/LayerBase.h"
#include "Tensors/Shape.h"
#include "Tools.h"
#include "Models/ModelBase.h"
#include "Types.h"

namespace Neuro
{
	map<string, int> LayerBase::s_LayersCountPerType;

	//////////////////////////////////////////////////////////////////////////
    LayerBase::LayerBase(const string& constructorName, const string& name)
	{
        m_ClassName = constructorName.substr(constructorName.find_last_of("::") + 1);
        m_Name = name.empty() ? GenerateName() : name;
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase* LayerBase::Clone()
	{
		Init(); // make sure parameter matrices are created
		LayerBase* clone = GetCloneInstance();
		clone->OnClone(*this);
		return clone;
	}

	//////////////////////////////////////////////////////////////////////////
	void LayerBase::OnClone(const LayerBase& source)
	{
		m_Name = source.m_Name;
	}

    //////////////////////////////////////////////////////////////////////////
    void LayerBase::LinkInput(LayerBase* inputLayer)
    {
        OnLink(inputLayer, true);
        inputLayer->OnLink(this, false);
    }

    ////////////////////////////////////////////////////////////////////////////
    //void LayerBase::OnLink(const vector<LayerBase*>& layers, bool input)
    //{
    //    if (input)
    //    {
    //        auto& inputShapes = InputShapes();

    //        inputShapes.clear();
    //        for (auto inLayer : layers)
    //            inputShapes.push_back(inLayer->OutputShape());
    //        InputLayers() = layers;
    //    }
    //    else
    //        assert(false);
    //}

    //////////////////////////////////////////////////////////////////////////
    /*void LayerBase::Link(const vector<LayerBase*>& inputLayers)
    {
        assert(!HasInputLayers());

        InputLayers().insert(InputLayers().end(), inputLayers.begin(), inputLayers.end());
        InputShapes().clear();

        OnLink(inputLayers, true);

        for (auto inLayer : inputLayers)
        {
            assert(inLayer->OutputShapes().size() == 1);
            inLayer->OnLink(this, false);            
        }
    }*/

    //////////////////////////////////////////////////////////////////////////
	void LayerBase::CopyParametersTo(LayerBase& target, float tau) const
	{
		assert(InputShape() == target.InputShape() && OutputShapes() == target.OutputShapes() && "Cannot copy parameters between incompatible layers.");
	}

	//////////////////////////////////////////////////////////////////////////
    const tensor_ptr_vec_t& LayerBase::FeedForward(const Tensor* input, bool training)
	{
        const_tensor_ptr_vec_t inputs;
        inputs.push_back(input);
        return FeedForward(inputs, training);
	}

	//////////////////////////////////////////////////////////////////////////
	void LayerBase::Init()
	{
		if (Initialized)
			return;

		OnInit();
		Initialized = true;
	}

    //////////////////////////////////////////////////////////////////////////
    vector<Tensor*> LayerBase::GetParams()
    {
        vector<Tensor*> params;
        vector<ParametersAndGradients> paramsAndGrads;
        GetParametersAndGradients(paramsAndGrads, false);
        for (auto i = 0; i < paramsAndGrads.size(); ++i)
            params.push_back(paramsAndGrads[i].Parameters);

        return params;
    }

    //////////////////////////////////////////////////////////////////////////
    bool LayerBase::CanStopBackProp() const
    {
        if (m_Trainable)
            return false;

        auto& inputLayers = InputLayers();

        if (inputLayers.empty())
            return true;

        for (auto inputLayer : inputLayers)
        {
            if (!inputLayer->CanStopBackProp())
                return false;
        }

        return true;
    }

    //////////////////////////////////////////////////////////////////////////
	string LayerBase::GenerateName() const
	{
        string classNameLower = ToLower(m_ClassName);
        stringstream ss;
		ss << classNameLower << "_" << (++s_LayersCountPerType[classNameLower]);
		return ss.str();
	}
}
