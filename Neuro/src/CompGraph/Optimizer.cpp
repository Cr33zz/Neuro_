#include <unordered_set>
#include <list>

#include "CompGraph/Optimizer.h"
#include "CompGraph/Variable.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    map<NodeBase*, Tensor*> Optimizer::ComputeGradients(NodeBase* lossNode)
    {
        map<NodeBase*, Tensor*> gradTable;
        gradTable[lossNode] = &(new Tensor(lossNode->m_Output.GetShape()))->FillWithValue(1);

        unordered_set<NodeBase*> visited;
        list<NodeBase*> queue;

        visited.insert(lossNode);
        queue.push_back(lossNode);

        while (!queue.empty())
        {
            auto node = queue.front();
            queue.pop_front();

            if (node != lossNode)
            {
                auto nodeGrad = gradTable[node];
                nodeGrad->Zero(); // reset gradient

                for (auto consumer : node->m_Consumers)
                {
                    auto outputGrad = gradTable[consumer];
                    auto& inputsGrad = static_cast<Operation*>(consumer)->ComputeGradient(*outputGrad);

                    if (inputsGrad.size() == 1)
                    {
                        nodeGrad->Add(*inputsGrad[0], *nodeGrad);
                    }
                    else
                    {
                        auto nodeIndexInConsumerInputs = distance(consumer->m_InputNodes.begin(), find(consumer->m_InputNodes.begin(), consumer->m_InputNodes.end(), node));
                        auto lossGradWrtNode = inputsGrad[nodeIndexInConsumerInputs];
                        nodeGrad->Add(*lossGradWrtNode, *nodeGrad);
                    }
                }
            }

            for (auto inputNode : node->m_InputNodes)
            {
                if (visited.find(inputNode) != visited.end())
                    continue;

                visited.insert(inputNode);
                queue.push_back(inputNode);
            }
        }

        return gradTable;
    }

    //////////////////////////////////////////////////////////////////////////
    void _SGDOptimimizer::MinimizationOperation::ComputeInternal()
    {
        auto gradTable = Optimizer::ComputeGradients(m_InputNodes[0]);

        for (auto entry : gradTable)
        {
            if (Variable* v = dynamic_cast<Variable*>(entry.first))
            {
                auto grad = entry.second;
                v->Value().Sub(grad->Mul(m_LearningRate), v->Value());
            }
        }
    }

}