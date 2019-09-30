#include "ComputationalGraph/Session.h"
#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Variable.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    Session* Session::Default = new Session();

    //////////////////////////////////////////////////////////////////////////
    vector<Tensor*> Session::Run(const vector<NodeBase*>& fetches, const map<Placeholder*, Tensor*>& feeds)
    {
        for(auto node : BuildForwardGraph(fetches))
        {
            if (Placeholder* p = dynamic_cast<Placeholder*>(node))
                node->m_Output = *feeds.find(p)->second;
            else if (Variable* v = dynamic_cast<Variable*>(node))
                node->m_Output = v->Value();
            else
            {
                Operation* op = static_cast<Operation*>(node);
                auto inputs = op->GatherInputs();
                op->Compute(inputs);
            }
        }
        vector<Tensor*> result(fetches.size());
        for (size_t i = 0; i < fetches.size(); ++i)
            result[i] = fetches[i]->OutputPtr();
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    std::vector<NodeBase*> Session::BuildForwardGraph(const vector<NodeBase*>& endNodes)
    {
        vector<NodeBase*> result;
        for (auto node : endNodes)
            ProcessForwardNode(node, result);
        return result;
    }

    //////////////////////////////////////////////////////////////////////////
    void Session::ProcessForwardNode(NodeBase* node, vector<NodeBase*>& nodes)
    {
        if (node->IsOp())
        {
            Operation* op = static_cast<Operation*>(node);
            for (auto inputNode : op->m_InputNodes)
                ProcessForwardNode(inputNode, nodes);
        }
        nodes.push_back(node);
    }
}
