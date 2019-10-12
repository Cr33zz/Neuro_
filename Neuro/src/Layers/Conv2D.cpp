#include "Layers/Conv2D.h"
#include "Tensors/Tensor.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Ops.h"
#include "Activations.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
    Conv2D::Conv2D(uint32_t filtersNum, uint32_t filterSize, uint32_t stride, uint32_t padding, ActivationBase* activation, EDataFormat dataFormat, const string& name)
        : SingleLayer(__FUNCTION__, Shape(), activation, name)
    {
        m_FilterSize = filterSize;
        m_FiltersNum = filtersNum;
        m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
    }

	//////////////////////////////////////////////////////////////////////////
	Conv2D::Conv2D(const Shape& inputShape, uint32_t filtersNum, uint32_t filterSize, uint32_t stride, uint32_t padding, ActivationBase* activation, EDataFormat dataFormat, const string& name)
		: SingleLayer(__FUNCTION__, inputShape, activation, name)
	{
		m_FilterSize = filterSize;
		m_FiltersNum = filtersNum;
		m_Stride = stride;
        m_Padding = padding;
        m_DataFormat = dataFormat;
	}

    //////////////////////////////////////////////////////////////////////////
	Conv2D::~Conv2D()
	{
		delete m_KernelInitializer;
		delete m_BiasInitializer;
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase* Conv2D::GetCloneInstance() const
	{
		return new Conv2D();
	}

	//////////////////////////////////////////////////////////////////////////
	void Conv2D::OnClone(const LayerBase& source)
	{
		__super::OnClone(source);

		auto& sourceConv = static_cast<const Conv2D&>(source);
		m_Kernels = new Variable(*sourceConv.m_Kernels);
        m_Bias = new Variable(*sourceConv.m_Bias);
		m_UseBias = sourceConv.m_UseBias;
		m_FilterSize = sourceConv.m_FilterSize;
		m_FiltersNum = sourceConv.m_FiltersNum;
		m_Stride = sourceConv.m_Stride;
	}

    //////////////////////////////////////////////////////////////////////////
    void Conv2D::Build(const vector<Shape>& inputShapes)
    {
        if (m_DataFormat == NCHW)
        {
            m_Kernels = new Variable(Shape(m_FilterSize, m_FilterSize, inputShapes[0].Depth(), m_FiltersNum), m_KernelInitializer, "kernels");
            if (m_UseBias)
                m_Bias = new Variable(Shape(1, 1, m_FiltersNum), m_BiasInitializer, "bias");
        }
        else
        {
            m_Kernels = new Variable(Shape(m_FilterSize, m_FilterSize, inputShapes[0].Len(0), m_FiltersNum), m_KernelInitializer, "kernels");
            if (m_UseBias)
                m_Bias = new Variable(Shape(m_FiltersNum), m_BiasInitializer, "bias");
        }

        m_Built = true;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> Conv2D::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        TensorLike* output = conv2d(inputs[0], m_Kernels, m_Stride, m_Padding, m_DataFormat);
        if (m_UseBias)
            output = add(output, m_Bias);
        if (m_Activation)
            output = m_Activation->Build(output);
        return { output };
    }

    //////////////////////////////////////////////////////////////////////////
	void Conv2D::Parameters(vector<Variable*>& params, bool onlyTrainable) const
	{
        if (onlyTrainable && !m_Trainable)
            return;

        params.push_back(m_Kernels);

		if (m_UseBias)
            params.push_back(m_Bias);
	}

    //////////////////////////////////////////////////////////////////////////
    void Conv2D::SerializedParameters(vector<SerializedParameter>& params)
    {
        params.push_back({ m_Kernels, { DepthAxis, BatchAxis, HeightAxis, WidthAxis } });

        if (m_UseBias)
        {
            if (m_DataFormat == NCHW)
                params.push_back({ m_Bias, { DepthAxis, HeightAxis, WidthAxis } });
            else
                params.push_back({ m_Bias });
        }
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Tensor& Conv2D::Kernels()
    {
        return m_Kernels->Output();
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Tensor& Conv2D::Bias()
    {
        return m_Bias->Output();
    }

    //////////////////////////////////////////////////////////////////////////
	void Conv2D::CopyParametersTo(LayerBase& target, float tau) const
	{
		__super::CopyParametersTo(target, tau);

		auto& targetConv = static_cast<Conv2D&>(target);
		m_Kernels->Output().CopyTo(targetConv.m_Kernels->Output(), tau);
		m_Bias->Output().CopyTo(targetConv.m_Bias->Output(), tau);
	}

	//////////////////////////////////////////////////////////////////////////
    Conv2D* Conv2D::KernelInitializer(InitializerBase* initializer)
    {
        delete m_KernelInitializer;
        m_KernelInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2D* Conv2D::BiasInitializer(InitializerBase* initializer)
    {
        delete m_BiasInitializer;
        m_BiasInitializer = initializer;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    Conv2D* Conv2D::UseBias(bool useBias)
    {
        m_UseBias = useBias;
        return this;
    }
}
