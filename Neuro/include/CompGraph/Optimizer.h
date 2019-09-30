#pragma once

#include <map>
#include "CompGraph/Operation.h"

namespace Neuro
{
    class Operation;

    class Optimizer
    {
    public:
        virtual Operation* Minimize(NodeBase* lossNode) = 0;

        static map<NodeBase*, Tensor*> ComputeGradients(NodeBase* lossNode);
    };

    class _SGDOptimimizer : public Optimizer
    {
    public:
        virtual Operation* Minimize(NodeBase* lossNode)
        {
            return new MinimizationOperation(lossNode, 0.02f);
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
    };
}
