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
    __int64 Stopwatch::ElapsedMilliseconds() const
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
        return (int)(ElapsedMilliseconds() / 1000);
    }

    //////////////////////////////////////////////////////////////////////////
    __int64 Stopwatch::ElapsedMicrosecondsSinceStart() const
    {
        if (!m_IsRunning)
            return 0;

        return chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - m_StartTimestamp).count();
    }

    //////////////////////////////////////////////////////////////////////////
    AutoStopwatch::AutoStopwatch(EAutoStopwatchMode mode)
    {
        m_Mode = mode;
        m_Timer.Start();
    }

    //////////////////////////////////////////////////////////////////////////
    string AutoStopwatch::ToString() const
    {
        __int64 duration = 0;
        string unit = "";
        switch (m_Mode)
        {
        case Milliseconds:
            duration = m_Timer.ElapsedMilliseconds();
            unit = "ms";
            break;
        case Microseconds:
            duration = m_Timer.ElapsedMicroseconds();
            unit = "us";
            break;
        case Seconds:
            duration = m_Timer.ElapsedSeconds();
            unit = "s";
            break;
        }

        return to_string(duration) + unit;
    }
}