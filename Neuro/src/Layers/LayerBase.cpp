#include <sstream>

#include "Activations.h"
#include "Layers/LayerBase.h"
#include "Tensors/Shape.h"
#include "Tools.h"
#include "Models/ModelBase.h"
#include "ComputationalGraph/TensorLike.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/NameScope.h"

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
	void LayerBase::CopyParametersTo(LayerBase& target, float tau) const
	{
	}

    //////////////////////////////////////////////////////////////////////////
    void LayerBase::SetTrainable(bool trainable)
    {
        m_Trainable = trainable;

        vector<Variable*> params;

        Parameters(params, false);
        for (auto param : params)
            param->Trainable(trainable);
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> LayerBase::Init(const vector<TensorLike*>& inputNodes, TensorLike* training)
    {
        NameScope scope(Name());

        if (!m_Built)
        {
            CheckInputCompatibility(inputNodes);

            for_each(inputNodes.begin(), inputNodes.end(), [&](TensorLike* t) { m_InputShapes.push_back(t->GetShape()); });

            Build(m_InputShapes);
            m_Built = true;
        }

        CheckInputCompatibility(inputNodes);

        m_OutputNodes = InitOps(inputNodes, training);

        for (size_t i = 0; i < m_OutputNodes.size(); ++i)
        {
            auto outNode = m_OutputNodes[i];
            
            outNode->m_Origin = new TensorLike::origin{ this, i };
            m_OutputShapes.push_back(outNode->GetShape());
        }

        return m_OutputNodes;
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
	string LayerBase::GenerateName() const
	{
        string classNameLower = ToLower(m_ClassName);
        stringstream ss;
		ss << classNameLower << "_" << (++s_LayersCountPerType[classNameLower]);
		return ss.str();
	}
}
