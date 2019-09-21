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

    // The concept of layer is that it is a 'block box' that supports forward and backward propagation.
    // Layer can have multiple inputs and outputs. Models are layers and can be combined with each other.
    // I.e. flow model (with single input and output) can be used as part of sequential model and be combined
    // with 'regular' layers.
    class LayerBase
    {
	public:
        virtual ~LayerBase() {}

        virtual const vector<Shape>& InputShapes() const = 0;
        virtual const vector<Tensor*>& InputsGradient() = 0;
        virtual const tensor_ptr_vec_t& Outputs() const = 0;
        virtual const vector<Shape>& OutputShapes() const = 0;
        virtual const vector<LayerBase*>& InputLayers() const = 0;
        virtual const vector<LayerBase*>& OutputLayers() const = 0;

		const Shape& InputShape() const { return InputShapes()[0]; }
		//const Tensor* InputGradient() { return InputsGradient()[0]; }
		const Tensor* Output() const { return Outputs()[0]; }
		const Shape& OutputShape() const { return OutputShapes()[0]; }
		const LayerBase* InputLayer() const { return InputLayers().empty() ? nullptr : InputLayers()[0]; }
        const LayerBase* OutputLayer() const { return OutputLayers().empty() ? nullptr : OutputLayers()[0]; }

        bool HasInputShape() const { return !InputShapes().empty();  }
        bool HasInputLayers() const { return !InputLayers().empty(); }

        // Return offset under which input layer related data can be found in input/inputGradient vectors.
        // It becomes useful in flow model where some layers can have multiple input layers and during back
        // propagation we need to figure out which subset of input gradients should be passed over to which
        // input layer.
        virtual int InputOffset(const LayerBase* inputLayer) const = 0;
        
        void LinkInput(LayerBase* inputLayer);

        // Tau specifies the percentage of copied parameters to be applied on a target network, when less than 1 target's network
        // parameters will be updated as follows: this_parameters * tau + target_parameters * (1 - tau)
        virtual void CopyParametersTo(LayerBase& target, float tau = 0) const;

        const tensor_ptr_vec_t& FeedForward(const Tensor* input, bool training);

        virtual const tensor_ptr_vec_t& FeedForward(const const_tensor_ptr_vec_t& inputs, bool training) = 0;
        virtual const tensor_ptr_vec_t& BackProp(const tensor_ptr_vec_t& outputsGradient) = 0;

        virtual void SetTrainable(bool trainable) { m_Trainable = trainable; }
        bool Trainable() const { return m_Trainable; }

        virtual uint32_t ParamsNum() const { return 0; }
        virtual void GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads, bool onlyTrainable = true) {}

		LayerBase* Clone();
		void Init();
		
        const string& ClassName() const { return m_ClassName; }

        vector<Tensor*> GetParams();

        const string& Name() const { return m_Name; }

	protected:
        
        LayerBase(const string& constructorName, const string& name = "");
		// This constructor exists only for cloning purposes
        LayerBase() {}

        virtual LayerBase* GetCloneInstance() const = 0;
        virtual void OnClone(const LayerBase& source);
        virtual void OnInit() {}
        virtual void OnLink(LayerBase* layer, bool input) = 0;
        
        bool CanStopBackProp() const;
        
		string GenerateName() const;

		bool m_Trainable = true;

	private:
		string m_Name;
        string m_ClassName;
		bool Initialized = false;

		static map<string, int> s_LayersCountPerType;

        friend class ModelBase;
		friend class Flow;
        friend class Sequential;
	};
}
