#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <list>

namespace Neuro
{
    using namespace std;

    class Tensor;
    class Placeholder;

    struct ILoader
    {
        virtual ~ILoader() {}
        virtual void operator()(Tensor& dest) = 0;
    };

    class DataPreloader
    {
    public:
        DataPreloader(const vector<Placeholder*>& destination, const vector<ILoader*>& loaders, size_t capacity);

        /*~DataPreloader()
        {
            m_Stop;
            m_PreloaderThread
        }*/

        // This function will copy first available tensors to the destination placeholders
        void Load();

    private:
        void Preload();

        thread m_PreloaderThread;

        condition_variable m_AvailableCond;
        mutex m_AvailableMtx;
        list<vector<Tensor>*> m_Available;
        condition_variable m_PendingCond;
        mutex m_PendingMtx;
        list<vector<Tensor>*> m_Pending;

        vector<Placeholder*> m_Destination;
        vector<ILoader*> m_Loaders;
    };
}
