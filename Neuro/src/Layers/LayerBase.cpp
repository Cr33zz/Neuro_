#include <sstream>

#include "Activations.h"
#include "Layers/LayerBase.h"
#include "Tensors/Shape.h"

namespace Neuro
{
	map<const char*, int> LayerBase::LayersCountPerType;

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(LayerBase* inputLayer, const Shape& outputShape, ActivationFunc* activation, const string& name)
		: LayerBase(outputShape, activation, name)
	{
		InputShapes.push_back(inputLayer->OutputShape);
		InputLayers.push_back(inputLayer);
		inputLayer->OutputLayers.push_back(this);
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const vector<LayerBase*>& inputLayers, const Shape& outputShape, ActivationFunc* activation, const string& name)
		: LayerBase(outputShape, activation, name)
	{
		InputLayers.insert(InputLayers.end(), inputLayers.begin(), inputLayers.end());
		for (auto inLayer : inputLayers)
		{
			InputShapes.push_back(inLayer->OutputShape);
			inLayer->OutputLayers.push_back(this);
		}
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const Shape& inputShape, const Shape& outputShape, ActivationFunc* activation, const string& name)
		: LayerBase(outputShape, activation, name)
	{
		InputShapes.push_back(inputShape);
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const vector<Shape>& inputShapes, const Shape& outputShape, ActivationFunc* activation, const string& name)
		: LayerBase(outputShape, activation, name)
	{
		InputShapes = inputShapes;
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase(const Shape& outputShape, ActivationFunc* activation, const string& name)
	{
		OutputShape = outputShape;
		Activation = activation;
		Name = name;
	}

	//////////////////////////////////////////////////////////////////////////
	LayerBase::LayerBase()
	{

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
		InputShapes = source.InputShapes;
		OutputShape = source.OutputShape;
		Activation = source.Activation;
		Name = source.Name;
	}

	//////////////////////////////////////////////////////////////////////////
	void LayerBase::CopyParametersTo(LayerBase& target, float tau)
	{
		/*if (InputShapes != target.InputShapes || OutputShape != target.OutputShape)
			throw new Exception("Cannot copy parameters between incompatible layers.");*/
	}

	//////////////////////////////////////////////////////////////////////////
	void LayerBase::ExecuteFeedForward()
	{
		Shape outShape(OutputShape.Width(), OutputShape.Height(), OutputShape.Depth(), Inputs[0]->BatchSize());
		// shape comparison is required for cases when last batch has different size
		if (Output.GetShape() != outShape)
			Output = Tensor(outShape);

		//FeedForwardTimer.Start();
		FeedForwardInternal();
		//FeedForwardTimer.Stop();

		if (Activation)
		{
			//ActivationTimer.Start();
			Activation->Compute(Output, Output);
			//ActivationTimer.Stop();

			/*if (NeuralNetwork::DebugMode)
				Trace.WriteLine($"Activation({Activation.GetType().Name}) output:\n{Output}\n");*/
		}
	}

	//////////////////////////////////////////////////////////////////////////
	const Tensor* LayerBase::FeedForward(const Tensor* input)
	{
		if (!Initialized)
			Init();

		Inputs.resize(1);
		Inputs[0] = input;
		ExecuteFeedForward();
		return &Output;
	}

	//////////////////////////////////////////////////////////////////////////
	const Tensor* LayerBase::FeedForward(const vector<const Tensor*>& inputs)
	{
		if (!Initialized)
			Init();

		Inputs = inputs;
		ExecuteFeedForward();
		return &Output;
	}

	//////////////////////////////////////////////////////////////////////////
	vector<Tensor>& LayerBase::BackProp(Tensor& outputGradient)
	{
		InputsGradient.resize(InputShapes.size());

		for (int i = 0; i < (int)InputShapes.size(); ++i)
		{
			auto& inputShape = InputShapes[i];
			Shape deltaShape(inputShape.Width(), inputShape.Height(), inputShape.Depth(), outputGradient.BatchSize());
			if (InputsGradient[i].GetShape() != deltaShape)
				InputsGradient[i] = Tensor(deltaShape);
		}

		// apply derivative of our activation function to the errors computed by previous layer
		if (Activation)
		{
			//ActivationBackPropTimer.Start();
			Activation->Derivative(Output, outputGradient, outputGradient);
			//ActivationBackPropTimer.Stop();

			/*if (NeuralNetwork::DebugMode)
				Trace.WriteLine($"Activation({Activation.GetType().Name}) errors gradient:\n{outputGradient}\n");*/
		}

		//BackPropTimer.Start();
		BackPropInternal(outputGradient);
		//BackPropTimer.Stop();

		return InputsGradient;
	}

	//////////////////////////////////////////////////////////////////////////
	int LayerBase::GetParamsNum()
	{
		return 0;
	}

	//////////////////////////////////////////////////////////////////////////
	void LayerBase::GetParametersAndGradients(vector<ParametersAndGradients>& result)
	{
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
	void LayerBase::OnInit()
	{
	}

	//////////////////////////////////////////////////////////////////////////
	string LayerBase::GenerateName() const
	{
		stringstream ss;
		ss << ClassName() << "_" << (++LayersCountPerType[ClassName()]);
		return ss.str();
	}
}
