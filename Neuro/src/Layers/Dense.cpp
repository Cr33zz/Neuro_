#include "Layers/Dense.h"
#include "Tensors/Tensor.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Zeros.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Dense::Dense(LayerBase* inputLayer, int outputs, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, inputLayer, Shape(outputs), activation, name)
	{
	}

    //////////////////////////////////////////////////////////////////////////
    Dense::Dense(int outputs, ActivationBase* activation, const string& name)
        : SingleLayer(__FUNCTION__, Shape(outputs), activation, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
	Dense::Dense(int inputs, int outputs, ActivationBase* activation, const string& name)
		: SingleLayer(__FUNCTION__, Shape(inputs), Shape(outputs), activation, name)
	{
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
	void Dense::InternalCall(TensorLike* training, bool initValues)
	{
        m_Weights = new Variable(Shape(OutputShape().Length, InputShape().Length), initValues ? m_WeightsInitializer : nullptr, "weights");
        m_Bias = new Variable(OutputShape(), initValues ? m_BiasInitializer : nullptr, "bias");

        m_OutputNodes[0] = matmul(m_InputNodes[0], m_Weights);
        if (m_UseBias)
            m_OutputNodes[0] = add(m_OutputNodes[0], m_Bias);
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
		return InputShape().Length * OutputShape().Length + (m_UseBias ? OutputShape().Length : 0);
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
