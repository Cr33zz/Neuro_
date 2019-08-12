#include "ChartGenerator.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    ChartGenerator::ChartGenerator(const string& outputFileName, const string& title, const string& xAxisLabel)
    {
        /*OutputFileName = $"{outputFileName}_{DateTime.Now.ToString("ddMMyyy_HHmm")}";
        ChartArea.AxisX.Title = xAxisLabel;
        ChartArea.AxisX.TitleFont = new Font(Chart.Font.Name, 11);
        ChartArea.AxisX.LabelStyle.Format = "#";
        ChartArea.AxisX.IsStartedFromZero = false;
        ChartArea.AxisY2.IsStartedFromZero = false;
        ChartArea.AxisY.IsStartedFromZero = false;
        Chart.ChartAreas.Add(ChartArea);
        Chart.Legends.Add(Legend);
        Legend.Font = new Font(Chart.Font.Name, 11);
        Chart.Width = 1000;
        Chart.Height = 600;
        Chart.Titles.Add(new Title(title, Docking.Top, new Font(Chart.Font.Name, 13), Color.Black));*/
    }

    //////////////////////////////////////////////////////////////////////////
    void ChartGenerator::AddSeries(int id, const string& label, int color, bool useSecondaryAxis)
    {
        /*Series s = new Series(id.ToString());
        s.ChartType = SeriesChartType.Line;
        s.BorderWidth = 2;
        s.Color = color;
        s.LegendText = label;
        s.Legend = "leg";
        s.IsVisibleInLegend = true;
        Chart.Series.Add(s);

        if (useSecondaryAxis)
        {
            ChartArea.AxisY2.Enabled = AxisEnabled.True;
            s.YAxisType = AxisType.Secondary;
        }*/
    }

    //////////////////////////////////////////////////////////////////////////
    void ChartGenerator::AddData(float x, float h, int seriesId)
    {
        /*if (Chart.Series.IndexOf(seriesId.ToString()) == -1)
            return;

        Chart.Series[seriesId.ToString()].Points.AddXY(x, h);
        DataMinX = Math.Min(DataMinX, x);
        DataMaxX = Math.Max(DataMaxX, x);*/
    }

    //////////////////////////////////////////////////////////////////////////
    void ChartGenerator::Save()
    {
        /*ChartArea.AxisX.Minimum = DataMinX;
        ChartArea.AxisX.Maximum = DataMaxX;
        Chart.SaveImage($"{OutputFileName}.png", ChartImageFormat.Png);*/
    }
}

