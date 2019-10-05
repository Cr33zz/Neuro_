#include <sstream>

#include "Activations.h"
#include "Layers/LayerBase.h"
#include "Tensors/Shape.h"
#include "Tools.h"
#include "Models/ModelBase.h"
#include "ComputationalGraph/Variable.h"

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
    LayerBase* LayerBase::LinkImpl(const vector<LayerBase*>& inputLayers)
    {
        OnLinkInput(inputLayers);
        for (auto inputLayer : inputLayers)
        {
            // linking to model layer is not allowed when there are multiple model output layers;
            // in that case specific model output layer(s) should be used to link
            assert(dynamic_cast<ModelBase*>(inputLayer) == nullptr || static_cast<ModelBase*>(inputLayer)->ModelOutputLayers().size() == 1);
            inputLayer->OnLinkOutput(this);
        }
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    void LayerBase::SerializedParameters(vector<SerializedParameter>& serializedParams)
    {
        vector<Variable*> params;
        Parameters(params, false);

        for (auto param : params)
            serializedParams.push_back({ param });
    }

    //////////////////////////////////////////////////////////////////////////
	LayerBase* LayerBase::Clone()
	{
		//Init(); // make sure parameter matrices are created
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
    LayerBase* LayerBase::Link(const vector<LayerBase*>& inputLayers)
    {
        return LinkImpl(inputLayers);
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* LayerBase::operator()(const vector<LayerBase*>& inputLayers)
    {
        Link(inputLayers);
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* LayerBase::Link(LayerBase* inputLayer)
    {
        return Link(vector<LayerBase*>{ inputLayer });
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* LayerBase::operator()(LayerBase* inputLayer)
    {
        Link(inputLayer);
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
	void LayerBase::CopyParametersTo(LayerBase& target, float tau) const
	{
		assert(InputShape() == target.InputShape() && OutputShapes() == target.OutputShapes() && "Cannot copy parameters between incompatible layers.");
	}

    //////////////////////////////////////////////////////////////////////////
    void LayerBase::SetTrainable(bool trainable)
    {

        m_Trainable = trainable;
    }

    //////////////////////////////////////////////////////////////////////////
	void LayerBase::Init(TensorLike* training, bool initValues)
	{
		if (Initialized)
			return;

		OnInit(training, initValues);
		Initialized = true;
	}

    //////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t LayerBase::Weights()
    {
        tensor_ptr_vec_t weights;
        vector<Variable*> params;

        Parameters(params, false);
        for (auto param : params)
            weights.push_back(param->OutputPtr());

        return weights;
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
