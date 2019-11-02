#pragma once

#include <vector>
#include <map>

namespace Neuro
{
    using namespace std;

    class TensorLike;
    class Operation;
    class Placeholder;
    class Tensor;
    class Variable;
    class Graph;

    class Session
    {
    public:
        Session(Graph* graph = nullptr);

        static Session* Default();
        static size_t GetFetchesHash(const vector<TensorLike*>& fetches);

        vector<Tensor*> Run(const vector<TensorLike*>& fetches, const map<Placeholder*, const Tensor*>& feeds = {});
        vector<Tensor*> RunInOrder(const vector<TensorLike*>& order, const vector<TensorLike*>& fetches, const map<Placeholder*, const Tensor*>& feeds);

    private:
        Graph* m_Graph;
        map<size_t, vector<TensorLike*>> m_OrderCache;
        static Session* s_Default;
    };
}
