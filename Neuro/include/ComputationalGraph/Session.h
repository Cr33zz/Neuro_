#pragma once

#include <vector>
#include <map>

namespace Neuro
{
    using namespace std;

    class NodeBase;
    class Operation;
    class Placeholder;
    class Tensor;

    class Session
    {
    public:
        static Session* Default;

        vector<Tensor*> Run(const vector<NodeBase*>& fetches, const map<Placeholder*, Tensor*>& feeds);

    private:
        vector<NodeBase*> BuildForwardGraph(const vector<NodeBase*>& endNodes);

        void ProcessForwardNode(NodeBase* node, vector<NodeBase*>& nodes);
    };
}
