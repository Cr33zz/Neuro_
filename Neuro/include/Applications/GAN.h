#pragma once

#include "Types.h"
#include "Tensors/Shape.h"

namespace Neuro
{
    class ModelBase;
    class Tensor;
    class TensorLike;

    struct GAN
    {
        static ModelBase* CreateUNetGenerator(const Shape& imgShape, uint32_t filtersStart = 64);
        static ModelBase* CreatePatchDiscriminator(const Shape& imgShape, uint32_t filtersStart = 64);
    };
}
