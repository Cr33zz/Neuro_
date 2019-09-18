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

		const Shape& InputShape() const { return InputShapes()[0]; }
        virtual const vector<Shape>& InputShapes() const = 0;
        const Tensor* Input() { return Inputs().empty() ? nullptr : Inputs()[0]; }
        virtual const tensor_ptr_vec_t& Inputs() const = 0;
		Tensor& InputGradient() { return InputsGradient()[0]; }
        virtual vector<Tensor>& InputsGradient() = 0;
		const Tensor& Output() const { return Outputs()[0]; }
        virtual const vector<Tensor>& Outputs() const = 0;
		const Shape& OutputShape() const { return OutputShapes()[0]; }
        virtual const vector<Shape>& OutputShapes() const = 0;
		virtual const ActivationBase* Activation() const { return nullptr; }
        const LayerBase* InputLayer() const { return InputLayers().empty() ? nullptr : InputLayers()[0]; }
        virtual const vector<LayerBase*>& InputLayers() const = 0;
        const LayerBase* OutputLayer() const { return OutputLayers().empty() ? nullptr : OutputLayers()[0]; }
        virtual const vector<LayerBase*>& OutputLayers() const = 0;

        bool HasInputShape() const { return !InputShapes().empty();  }
        bool HasInputLayers() const { return !InputLayers().empty(); }

		const string& Name() const { return m_Name; }

        void Link(LayerBase* inputLayer);
        void Link(const vector<LayerBase*>& inputLayers);

        // Tau specifies the percentage of copied parameters to be applied on a target network, when less than 1 target's network
        // parameters will be updated as follows: this_parameters * tau + target_parameters * (1 - tau)
        virtual void CopyParametersTo(LayerBase& target, float tau = 0) const;

		const Tensor& FeedForward(const Tensor* input, bool training);
		const Tensor& FeedForward(const tensor_ptr_vec_t& inputs, bool training);
		vector<Tensor>& BackProp(vector<Tensor>& outputsGradient);

        virtual void SetTrainable(bool trainable) { m_Trainable = trainable; }
        bool Trainable() const { return m_Trainable; }

        virtual uint32_t ParamsNum() const { return 0; }
        virtual void GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads, bool onlyTrainable = true) {}

		LayerBase* Clone();
		void Init();
		
        const string& ClassName() const { return m_ClassName; }

        int FeedForwardTime() const { return (int)m_FeedForwardTimer.ElapsedMilliseconds(); }
        int BackPropTime() const { return (int)m_BackPropTimer.ElapsedMilliseconds(); }
        int ActivationTime() const { return (int)m_ActivationTimer.ElapsedMilliseconds(); }
        int ActivationBackPropTime() const { return (int)m_ActivationBackPropTimer.ElapsedMilliseconds(); }

        vector<Tensor*> GetParams();

	protected:
        // The concept of layer is that it is a 'block box' that supports forward and backward propagation.
        // Feed forward: input Tensor -> |logic| -> output Tensor
        // Back propagation: output gradients -> parameters and inputs gradients (for predecessing layer outputs)
        // These error gradients are always of the same size as respective outputs and are saying now much each output
        // contributed to the final error)
        LayerBase(const string& constructorName, const string& name = "");
		// This constructor exists only for cloning purposes
        LayerBase() {}

        virtual vector<Shape>& InputShapes() = 0;
        virtual tensor_ptr_vec_t& Inputs() = 0;
        virtual vector<Tensor>& Outputs() = 0;
        virtual vector<Shape>& OutputShapes() = 0;
        virtual vector<LayerBase*>& InputLayers() = 0;
        virtual vector<LayerBase*>& OutputLayers() = 0;

        virtual LayerBase* GetCloneInstance() const = 0;
        virtual void OnClone(const LayerBase& source);
        virtual void OnInit() {}
        virtual void OnLink(LayerBase* layer, bool input);
        virtual void OnLink(const vector<LayerBase*>& layers, bool input);
        
        virtual void FeedForwardInternal(bool training) {}

        // Overall implementation of back propagation should look like this:
        // - if there is activation function apply derivative of that function to the errors computed by previous layer Errors.MultElementWise(Output.Map(x => ActivationF(x, true)));
        // - update errors in next layer (how much each input contributes to our output errors in relation to our parameters) stored InputDelta
        // - update parameters using error and input
        virtual void BackPropInternal(vector<Tensor>& outputsGradient) {}

        bool CanStopBackProp() const;
        
		string GenerateName() const;

		bool m_Trainable = true;

	private:
		void ExecuteFeedForward(bool training);
		
		string m_Name;
        string m_ClassName;
		bool Initialized = false;

		Stopwatch m_FeedForwardTimer;
		Stopwatch m_ActivationTimer;
		Stopwatch m_BackPropTimer;
		Stopwatch m_ActivationBackPropTimer;

		static map<string, int> s_LayersCountPerType;

        friend class ModelBase;
		friend class Flow;
        friend class Sequential;
	};
}
