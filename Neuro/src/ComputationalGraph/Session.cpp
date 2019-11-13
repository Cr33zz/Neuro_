#include "ComputationalGraph/Session.h"
#include "ComputationalGraph/Graph.h"
#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Variable.h"
#include "Tensors/Tensor.h"
#include "Tools.h"
#include "Debug.h"

//#define ENABLE_SESSION_LOGS

#ifdef ENABLE_SESSION_LOGS
#include <windows.h>
#include <debugapi.h>
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
    size_t Session::GetFetchesHash(const vector<TensorLike*>& fetches)
    {
        size_t fetchesHash = 0;
        std::hash<TensorLike*> hasher;
        for (size_t i = 0; i < fetches.size(); ++i)
            fetchesHash = fetchesHash * 31 + hasher(fetches[i]);
        return fetchesHash;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<Tensor*> Session::Run(const vector<TensorLike*>& fetches, const map<Placeholder*, const Tensor*>& feeds)
    {
        size_t fetchesHash = GetFetchesHash(fetches);
        auto orderIt = m_OrderCache.find(fetchesHash);
        if (orderIt == m_OrderCache.end())
        {
            m_OrderCache[fetchesHash] = m_Graph->BuildForwardOrder(fetches);
            orderIt = m_OrderCache.find(fetchesHash);
        }

        return RunInOrder(orderIt->second, fetches, feeds);
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
            NEURO_ASSERT(feed.second->GetShape() == feed.first->m_Output.GetShape(), "Mismatched feed shape. Expected: " << feed.first->m_Output.GetShape().ToString() << " received: " << feed.second->GetShape().ToString());
            feed.second->CopyTo(feed.first->m_Output);
        }

        for (size_t i = 0; i < fetches.size(); ++i)
            fetches[i]->Output().ResetRef(1); // lock fetches outputs so they don't get released

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
                auto inputs = op->GatherInputs();
                op->Compute(inputs);

                if (Debug::ShouldLogOutput(node->Name()))
                {
                    for (size_t i = 0; i < inputs.size(); ++i)
                        inputs[i]->DebugDumpValues(node->Name() + "_input" + to_string(i) + "_step" + to_string(Debug::GetStep()) + ".log");
                }
            }

            if (Debug::ShouldLogOutput(node->Name()))
                node->Output().DebugDumpValues(node->Name() + "_output0_step" + to_string(Debug::GetStep()) + ".log");
        }

        Debug::Step();

        vector<Tensor*> result(fetches.size());
        for (size_t i = 0; i < fetches.size(); ++i)
            result[i] = fetches[i]->OutputPtr();
        return result;
    }
}
