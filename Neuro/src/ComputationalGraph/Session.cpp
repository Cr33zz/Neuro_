#include "ComputationalGraph/Session.h"
#include "ComputationalGraph/Graph.h"
#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Variable.h"
#include "Tensors/Tensor.h"

//#define ENABLE_SESSION_LOGS

#ifdef ENABLE_SESSION_LOGS
#include <windows.h>
#include <debugapi.h>
#include "Tools.h"
#define SESSION_DEBUG_INFO(...) do { OutputDebugString(StringFormat(__VA_ARGS__).c_str()); } while(0)
#else
#define SESSION_DEBUG_INFO(...) {}
#endif

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
        {
            SESSION_DEBUG_INFO("##Session: Feeding '%s'...\n", feed.first->Name().c_str());
            feed.first->m_Output.ResizeBatch(feed.second->Batch());
            feed.second->CopyTo(feed.first->m_Output);
        }

        for (size_t n = 0; n < order.size(); ++n)
        {
            if (n + 1 < order.size())
            {
                auto node = order[n + 1];
                SESSION_DEBUG_INFO("##Session: Prefetching '%s'...\n", node->Name().c_str());
                node->Prefetch();
            }

            auto node = order[n];
            if (node->IsOp())
            {
                SESSION_DEBUG_INFO("##Session: Computing '%s'...\n", node->Name().c_str());
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
