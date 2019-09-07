#pragma once

#include <string>
#include <vector>
#include <map>

#include "Tensors/Tensor.h"
#include "Tensors/Shape.h"
#include "ParametersAndGradients.h"
#include "Stopwatch.h"

namespace Neuro
{
	using namespace std;

	class ActivationBase;

    class LayerBase
    {
	public:
        virtual ~LayerBase() {}

		const Shape& InputShape() const { return m_InputShapes[0]; }
		const vector<Shape>& InputShapes() const { return m_InputShapes; }
		const Tensor* Input() { return m_Inputs[0]; }
		const tensor_ptr_vec_t& Inputs() const { return m_Inputs; }
		Tensor& InputGradient() { return m_InputGradients[0]; }
		vector<Tensor>& InputsGradient() { return m_InputGradients; }
		const Tensor& Output() const { return m_Outputs[0]; }
        const vector<Tensor>& Outputs() const { return m_Outputs; }
		const Shape& OutputShape() const { return m_OutputShapes[0]; }
        const vector<Shape>& OutputShapes() const { return m_OutputShapes; }
		const ActivationBase* Activation() const { return m_Activation; }
        const LayerBase* InputLayer() const { return m_InputLayers[0]; }
        const vector<LayerBase*>& InputLayers() const { return m_InputLayers; }
        const LayerBase* OutputLayer() const { return m_OutputLayers[0]; }
		const vector<LayerBase*>& OutputLayers() const { return m_OutputLayers; }

        bool HasInputShape() const { return !m_InputShapes.empty();  }

		const string& Name() const { return m_Name; }

        void Link(LayerBase* inputLayer);
        void Link(const vector<LayerBase*>& inputLayers);

        virtual void CopyParametersTo(LayerBase& target, float tau = 0) const;

		const Tensor& FeedForward(const Tensor* input, bool training);
		const Tensor& FeedForward(const tensor_ptr_vec_t& inputs, bool training);
		vector<Tensor>& BackProp(vector<Tensor>& outputGradients);

        virtual uint32_t GetParamsNum() const { return 0; }
        virtual void GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads) {}

		LayerBase* Clone();
		void Init();
		
        const string& ClassName() const { return m_ClassName; }

        int FeedForwardTime() const { return (int)m_FeedForwardTimer.ElapsedMiliseconds(); }
        int BackPropTime() const { return (int)m_BackPropTimer.ElapsedMiliseconds(); }
        int ActivationTime() const { return (int)m_ActivationTimer.ElapsedMiliseconds(); }
        int ActivationBackPropTime() const { return (int)m_ActivationBackPropTimer.ElapsedMiliseconds(); }

	protected:
        // The concept of layer is that it is a 'block box' that supports forward and backward propagation.
        // Feed forward: input Tensor -> |logic| -> output Tensor
        // Back propagation: output gradients -> parameters and inputs gradients (for predecessing layer outputs)
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
        LayerBase() {}

        virtual LayerBase* GetCloneInstance() const = 0;
        virtual void OnClone(const LayerBase& source);
        virtual void OnInit() {}
        virtual void OnLink() {}
        
        virtual void FeedForwardInternal(bool training) {}

        // Overall implementation of back propagation should look like this:
        // - if there is activation function apply derivative of that function to the errors computed by previous layer Errors.MultElementWise(Output.Map(x => ActivationF(x, true)));
        // - update errors in next layer (how much each input contributes to our output errors in relation to our parameters) stored InputDelta
        // - update parameters using error and input
        virtual void BackPropInternal(vector<Tensor>& outputGradients) {}

        //virtual void SerializeParameters(XmlElement elem) {}
        //virtual void DeserializeParameters(XmlElement elem) {}
        
		string GenerateName() const;

		vector<const Tensor*> m_Inputs;
		vector<Shape> m_InputShapes;
		vector<Tensor> m_InputGradients;
        // Only models can have multiple outputs
		vector<Tensor> m_Outputs;
        // Only models can have multiple outputs shapes
		vector<Shape> m_OutputShapes;

	private:
		void ExecuteFeedForward(bool training);
		
		ActivationBase* m_Activation;
		vector<LayerBase*> m_InputLayers;
		vector<LayerBase*> m_OutputLayers;
		string m_Name;
        string m_ClassName;
		bool Initialized = false;        

		Stopwatch m_FeedForwardTimer;
		Stopwatch m_ActivationTimer;
		Stopwatch m_BackPropTimer;
		Stopwatch m_ActivationBackPropTimer;

		static map<string, int> s_LayersCountPerType;

		friend class Flow;
	};
}
