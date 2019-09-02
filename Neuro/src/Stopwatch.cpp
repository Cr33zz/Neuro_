#include "Stopwatch.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Stopwatch::Stopwatch()
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Stopwatch::Reset()
    {
        Stop();
        m_AccumulatedTime = 0;
    }

    //////////////////////////////////////////////////////////////////////////
    void Stopwatch::Restart()
    {
        Reset();
        Start();
    }

    //////////////////////////////////////////////////////////////////////////
    void Stopwatch::Start()
    {
        if (m_IsRunning)
            return;

        m_StartTimestamp = chrono::high_resolution_clock::now();
        m_IsRunning = true;
    }

    //////////////////////////////////////////////////////////////////////////
    void Stopwatch::Stop()
    {
        if (!m_IsRunning)
            return;

        m_AccumulatedTime += ElapsedMicrosecondsSinceStart();
        m_IsRunning = false;
    }

    //////////////////////////////////////////////////////////////////////////
    __int64 Stopwatch::ElapsedMiliseconds() const
    {
        return ElapsedMicroseconds() / 1000;
    }

    //////////////////////////////////////////////////////////////////////////
    __int64 Stopwatch::ElapsedMicroseconds() const
    {
        return m_AccumulatedTime + ElapsedMicrosecondsSinceStart();
    }

    //////////////////////////////////////////////////////////////////////////
    int Stopwatch::ElapsedSeconds() const
    {
        return (int)(ElapsedMiliseconds() / 1000);
    }

    //////////////////////////////////////////////////////////////////////////
    __int64 Stopwatch::ElapsedMicrosecondsSinceStart() const
    {
        if (!m_IsRunning)
            return 0;

        return chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - m_StartTimestamp).count();
    }
}