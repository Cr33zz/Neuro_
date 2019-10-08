#include "Layers/Dense.h"
#include "Tensors/Tensor.h"
#include "Activations.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Zeros.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
    Dense::Dense(uint32_t units, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, activation, name)
    {
        m_Units = units;
    }

    //////////////////////////////////////////////////////////////////////////
    Dense::Dense(uint32_t inputUnits, uint32_t units, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, Shape(inputUnits), activation, name)
    {
        m_Units = units;
    }

    //////////////////////////////////////////////////////////////////////////
	Dense::Dense()
	{
	}

    //////////////////////////////////////////////////////////////////////////
    Dense::~Dense()
	{
        delete m_WeightsInitializer;
		delete m_BiasInitializer;
	}

	//////////////////////////////////////////////////////////////////////////
	Neuro::LayerBase* Dense::GetCloneInstance() const
	{
		return new Dense();
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::OnClone(const LayerBase& source)
	{
		__super::OnClone(source);

		auto& sourceDense = static_cast<const Dense&>(source);
		m_Weights = new Variable(*sourceDense.m_Weights);
		m_Bias = new Variable(*sourceDense.m_Bias);
		m_UseBias = sourceDense.m_UseBias;
	}

    //////////////////////////////////////////////////////////////////////////
    void Dense::Build(const vector<Shape>& inputShapes)
    {
        NEURO_ASSERT(inputShapes.size() == 1, "Dense layer accepts single input.");
        NEURO_ASSERT(inputShapes[0].Batch() == 1, "");

        m_Weights = new Variable(Shape(m_Units, inputShapes[0].Length), m_WeightsInitializer, "weights");

        if (m_UseBias)
            m_Bias = new Variable(Shape(m_Units), m_BiasInitializer, "bias");

        m_Built = true;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Dense::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        NEURO_ASSERT(inputs.size() == 1, "Dense layer accepts single input.");

        TensorLike* output = matmul(inputs[0], m_Weights);
        if (m_UseBias)
            output = add(output, m_Bias);
        if (m_Activation)
            output = m_Activation->Build(output);
        return { output };
    }

    //////////////////////////////////////////////////////////////////////////
	void Dense::CopyParametersTo(LayerBase& target, float tau) const
	{
		__super::CopyParametersTo(target, tau);

		auto& targetDense = static_cast<Dense&>(target);
		m_Weights->Output().CopyTo(targetDense.m_Weights->Output(), tau);
		m_Bias->Output().CopyTo(targetDense.m_Bias->Output(), tau);
	}

	//////////////////////////////////////////////////////////////////////////
	uint32_t Dense::ParamsNum() const
	{
        return 0;//  InputShape().Length * OutputShape().Length + (m_UseBias ? OutputShape().Length : 0);
	}

	//////////////////////////////////////////////////////////////////////////
	void Dense::Parameters(vector<Variable*>& params, bool onlyTrainable)
	{
        if (onlyTrainable && !m_Trainable)
            return;

        params.push_back(m_Weights);

		if (m_UseBias)
            params.push_back(m_Bias);
	}

    //////////////////////////////////////////////////////////////////////////
    Dense* Dense::WeightsInitializer(InitializerBase* initializer)
    {
        delete m_WeightsInitializer;
        m_WeightsInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Dense* Dense::BiasInitializer(InitializerBase* initializer)
    {
        delete m_BiasInitializer;
        m_BiasInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Dense* Dense::UseBias(bool useBias)
    {
        m_UseBias = useBias;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Tensor& Dense::Weights()
    {
        return m_Weights->Output();
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Tensor& Dense::Bias()
    {
        return m_Bias->Output();
    }

}
