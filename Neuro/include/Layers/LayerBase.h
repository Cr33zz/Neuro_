#pragma once

#include <string>
#include <vector>
#include <map>

#include "Tensors/Tensor.h"
#include "Tensors/Shape.h"
#include "ParametersAndGradients.h"

namespace Neuro
{
	using namespace std;

	class ActivationBase;

    class LayerBase
    {
	public:
        virtual ~LayerBase() {}

		const vector<Shape>& InputShapes() const { return m_InputShapes; }
		const Shape& InputShape() const { return m_InputShapes[0]; }
		const tensor_ptr_vec_t& Inputs() const { return m_Inputs; }
		const Tensor* Input() { return m_Inputs[0]; }
		vector<Tensor>& InputsGradient() { return m_InputsGradient; }
		Tensor& InputGradient() { return m_InputsGradient[0]; }
		const Tensor& Output() const { return m_Output; }
		const Shape& OutputShape() const { return m_OutputShape; }
		const ActivationBase* Activation() const { return m_Activation; }
		const vector<LayerBase*>& InputLayers() const { return m_InputLayers; }
		const vector<LayerBase*>& OutputLayers() const { return m_OutputLayers; }
		const string& Name() const { return m_Name; }

        virtual void CopyParametersTo(LayerBase& target, float tau = 0) const;

		const Tensor* FeedForward(const Tensor* input, bool training);
		const Tensor* FeedForward(const vector<const Tensor*>& inputs, bool training);
		vector<Tensor>& BackProp(Tensor& outputGradient);

		virtual int GetParamsNum() const;
		virtual void GetParametersAndGradients(vector<ParametersAndGradients>& result);

		LayerBase* Clone();
		void Init();
		
        const string& ClassName() const { return m_ClassName; }

	protected:
        // The concept of layer is that it is a 'block box' that supports feed forward and backward propagation.
        // Feed forward: input Tensor -> |logic| -> output Tensor
        // Back propagation: error gradients (for its outputs) -> |learning| -> error gradients (for predecessing layer outputs) and internal parameters deltas
        // These error gradients are always of the same size as respective outputs and are saying now much each output
        // contributed to the final error)
        LayerBase(const string& constructorName, LayerBase* inputLayer, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
		LayerBase(const string& constructorName, const vector<LayerBase*>& inputLayers, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        // This constructor should only be used for input layer
        LayerBase(const string& constructorName, const Shape& inputShape, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        // This constructor should only be used for input layer
        LayerBase(const string& constructorName, const vector<Shape>& inputShapes, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
		LayerBase(const string& constructorName, const Shape& outputShape, ActivationBase* activation = nullptr, const string& name = "");
        // This constructor exists only for cloning purposes
        LayerBase();		

        virtual LayerBase* GetCloneInstance() const = 0;
        virtual void OnClone(const LayerBase& source);
		virtual void OnInit();

        virtual void FeedForwardInternal(bool training) = 0;

        // Overall implementation of back propagation should look like this:
        // - if there is activation function apply derivative of that function to the errors computed by previous layer Errors.MultElementWise(Output.Map(x => ActivationF(x, true)));
        // - update errors in next layer (how much each input contributes to our output errors in relation to our parameters) stored InputDelta
        // - update parameters using error and input
        virtual void BackPropInternal(Tensor& outputGradient) = 0;

        //virtual void SerializeParameters(XmlElement elem) {}
        //virtual void DeserializeParameters(XmlElement elem) {}
        
		string GenerateName() const;

		vector<const Tensor*> m_Inputs;
		vector<Shape> m_InputShapes;
		vector<Tensor> m_InputsGradient;
		Tensor m_Output;
		Shape m_OutputShape;

	private:
		void ExecuteFeedForward(bool training);
		
		ActivationBase* m_Activation;
		vector<LayerBase*> m_InputLayers;
		vector<LayerBase*> m_OutputLayers;
		string m_Name;
        string m_ClassName;
		bool Initialized = false;        

		/*Stopwatch FeedForwardTimer = new Stopwatch();
		Stopwatch ActivationTimer = new Stopwatch();
		Stopwatch BackPropTimer = new Stopwatch();
		Stopwatch ActivationBackPropTimer = new Stopwatch();*/

		static map<string, int> s_LayersCountPerType;

		friend class Flow;
	};
}
