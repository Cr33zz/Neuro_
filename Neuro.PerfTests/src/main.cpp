#include "SimpleNetPerfTests.h"
#include "TensorPerfTests.h"
#include "ConvNetPeftTests.h"
//networks
#include "IrisNetwork.h"

int main()
{
	//SimpleNetPerfTests::Run();
    //TensorPerfTests::Run();
    IrisNetwork::Run();

    return 0;
}