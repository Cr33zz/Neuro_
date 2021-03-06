#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

#include "Types.h"

namespace Neuro
{
    using namespace std;

    class Tensor;

    class TensorFormatter
    {
    public:
        static string ToString(const Tensor& t);

    private:
        class Dragon4FloatFormatter
        {
        public:
            Dragon4FloatFormatter(const float* a, size_t size);
            string FormatValue(float v) const;

        private:
            string Float2Str(float f, int precision = 5) const;

            size_t m_PadLeft = 0;
            size_t m_PadRight = 0;
        };

    private:
        static string ToStringRecursive(const Tensor& t, const vector<int>& index, const string& hanging_indent, int curr_width, const Dragon4FloatFormatter& formatter);
        static pair<string, string> ExtendLine(string& str, string& line, const string& word, int line_width, const string& next_line_prefix);
    };
}
