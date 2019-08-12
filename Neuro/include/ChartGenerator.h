#pragma once

#include <string>
#include <numeric>

namespace Neuro
{
    using namespace std;

    class ChartGenerator
    {
    public:
        ChartGenerator(const string& outputFileName, const string& title = "", const string& xAxisLabel = "");

        void AddSeries(int id, const string& label, int color, bool useSecondaryAxis = false);
        void AddData(float x, float h, int seriesId);
        void Save();

    private:
        //private Chart Chart = new Chart();
        //private ChartArea ChartArea = new ChartArea();
        //private Legend Legend = new Legend("leg");
        float m_DataMinX = numeric_limits<float>().max();
        float m_DataMaxX = numeric_limits<float>().min();
        string m_OutputFileName;
    };
}
