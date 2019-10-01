#pragma once

#include <map>
#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class Operation;
    class Variable;

    class Optimizer
    {
    public:
        virtual Operation* Minimize(NodeBase* loss) = 0;
        //virtual Operation* Maximize(NodeBase* loss) = 0;

        static vector<Variable*> ComputeGradients(NodeBase* loss);
    };

    class _SGDOptimizer : public Optimizer
    {
    public:
        _SGDOptimizer(float lr) : m_LearningRate(lr) {}

        virtual Operation* Minimize(NodeBase* loss)
        {
            return new MinimizationOperation(loss, 0.02f);
        }

    private:
        class MinimizationOperation : public Operation
        {
        public:
            MinimizationOperation(NodeBase* loss, float lr) :Operation({ loss }), m_LearningRate(lr) {}
        protected:
            virtual void ComputeInternal();

            virtual void ComputeGradientInternal(const Tensor& grad) {}

            float m_LearningRate;
        };

        float m_LearningRate;
    };
}
