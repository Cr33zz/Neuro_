#pragma once

#include "Types.h"
#include "Activations.h"
#include "Loss.h"
#include "Random.h"
#include "Tools.h"
#include "TestTools.h"
#include "Stopwatch.h"

#include "Layers/LayerBase.h"
//#include "Layers/Activation.h"
//#include "Layers/Conv2D.h"
//#include "Layers/Conv2DTranspose.h"
#include "Layers/Dense.h"
//#include "Layers/Flatten.h"
//#include "Layers/Reshape.h"
//#include "Layers/Dropout.h"
//#include "Layers/BatchNormalization.h"
#include "Layers/Input.h"
//#include "Layers/Pooling2D.h"
//#include "Layers/UpSampling2D.h"
//#include "Layers/Concatenate.h"
//#include "Layers/Merge.h"

#include "Models/ModelBase.h"
#include "Models/Sequential.h"
#include "Models/Flow.h"

#include "Optimizers/OptimizerBase.h"
#include "Optimizers/Adam.h"
#include "Optimizers/SGD.h"

#include "Initializers/InitializerBase.h"
#include "Initializers/Const.h"
#include "Initializers/GlorotNormal.h"
#include "Initializers/GlorotUniform.h"
#include "Initializers/Normal.h"
#include "Initializers/Uniform.h"
#include "Initializers/HeNormal.h"
#include "Initializers/HeUniform.h"
#include "Initializers/Zeros.h"

#include "Tensors/Shape.h"
#include "Tensors/Tensor.h"

#include "ComputationalGraph/TensorLike.h"
#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Graph.h"
#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Session.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Constant.h"

#include "ComputationalGraph/Ops.h"

#include "TestTools.h"