#include "DataPreloader.h"
#include "Tensors/Tensor.h"
#include "ComputationalGraph/Placeholder.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    DataPreloader::DataPreloader(const vector<Placeholder*>& destination, const vector<ILoader*>& loaders, size_t capacity)
        : m_Destination(destination), m_Loaders(loaders)
    {
        for (size_t i = 0; i < capacity; ++i)
        {
            vector<Tensor>* data = new vector<Tensor>(destination.size());
            for (size_t i = 0; i < destination.size(); ++i)
                (*data)[i].Resize(destination[i]->GetShape());
            m_Pending.push_back(data);
        }

        m_PreloaderThread = thread(&DataPreloader::Preload, this);
    }

    //////////////////////////////////////////////////////////////////////////
    void DataPreloader::Load()
    {
        vector<Tensor>* data = nullptr;
        {
            unique_lock<mutex> availableLocker(m_AvailableMtx);
            m_AvailableCond.wait(availableLocker, [this]() {return !m_Available.empty(); });

            data = m_Available.front();
            m_Available.pop_front();
        }

        // copy data to placeholders
        for (size_t i = 0; i < m_Destination.size(); ++i)
            (*data)[i].CopyTo(m_Destination[i]->Output());

        {
            unique_lock<mutex> pendingLocker(m_PendingMtx);
            m_Pending.push_back(data);
        }
        m_PendingCond.notify_all();
    }

    //////////////////////////////////////////////////////////////////////////
    void DataPreloader::Preload()
    {
        while (true)
        {
            vector<Tensor>* data = nullptr;

            {
                unique_lock<mutex> pendingLocker(m_PendingMtx);
                m_PendingCond.wait(pendingLocker, [this]() {return !m_Pending.empty(); });

                data = m_Pending.front();
                m_Pending.pop_front();
            }

            // load data
            for (size_t i = 0; i < m_Loaders.size(); ++i)
                (*m_Loaders[i])((*data)[i]);

            {
                unique_lock<mutex> availableLocker(m_AvailableMtx);
                m_Available.push_back(data);
            }
            m_AvailableCond.notify_all();
        }
    }

}