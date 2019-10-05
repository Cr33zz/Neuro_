#include "ComputationalGraph/Session.h"
#include "ComputationalGraph/Graph.h"
#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Variable.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    Session* Session::s_Default = nullptr;

    //////////////////////////////////////////////////////////////////////////
    Session::Session(Graph* graph)
    {
        if (!graph)
            graph = Graph::Default();

        m_Graph = graph;
    }

    //////////////////////////////////////////////////////////////////////////
    Session* Session::Default()
    {
        if (!s_Default)
            s_Default = new Session();

        return s_Default;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Tensor*> Session::Run(const vector<TensorLike*>& fetches, const map<Placeholder*, const Tensor*>& feeds)
    {
        return RunInOrder(m_Graph->BuildForwardOrder(fetches), fetches, feeds);
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Tensor*> Session::RunInOrder(const vector<TensorLike*>& order, const vector<TensorLike*>& fetches, const map<Placeholder*, const Tensor*>& feeds)
    {
        m_Graph->InitVariables();
        m_Graph->IncrementStep();

        for (auto feed : feeds)
            feed.first->m_Output = *feed.second;

        for (auto node : order)
        {
            if (node->IsOp())
            {
                Operation* op = static_cast<Operation*>(node);
                op->Compute(op->GatherInputs());
            }
        }

        m_Graph->DebugLog();

        vector<Tensor*> result(fetches.size());
        for (size_t i = 0; i < fetches.size(); ++i)
            result[i] = fetches[i]->OutputPtr();
        return result;
    }
}
