struct ProgramTestClass
{
	__device__  ProgramTestClass()
	{
	}
	int A;
	int B;
	int C;
};


// CudaTest.Program
extern "C" __global__ void doTheThing( ProgramTestClass* tests, int testsLen0);
// CudaTest.Program
extern "C" __global__ void addArrays( int* a, int aLen0,  int* b, int bLen0,  int* c, int cLen0);
// CudaTest.Program
extern "C" __global__ void findPrimes( int* toCheck, int toCheckLen0,  int* results, int resultsLen0);
// CudaTest.Program
__device__ int isPrime(int a);

// CudaTest.Program
extern "C" __global__ void doTheThing( ProgramTestClass* tests, int testsLen0)
{
	int x = blockIdx.x;
	if (x < 1000000)
	{
		ProgramTestClass testClass = tests[(x)];
		testClass.C = testClass.A + testClass.B;
	}
}
// CudaTest.Program
extern "C" __global__ void addArrays( int* a, int aLen0,  int* b, int bLen0,  int* c, int cLen0)
{
	int x = blockIdx.x;
	if (x < 1000000)
	{
		c[(x)] = a[(x)] + b[(x)];
	}
}
// CudaTest.Program
extern "C" __global__ void findPrimes( int* toCheck, int toCheckLen0,  int* results, int resultsLen0)
{
	int x = blockIdx.x;
	if (x < 1000000)
	{
		results[(x)] = isPrime(toCheck[(x)]);
	}
}
// CudaTest.Program
__device__ int isPrime(int a)
{
	int result;
	if (a == 1 || a == 2)
	{
		result = 1;
	}
	else
	{
		int num = a % 2;
		if (num == 0)
		{
			result = 0;
		}
		else
		{
			int num2 = a / 2;
			for (int num3 = 3; num3 <= num2; num3++)
			{
				if (a % num3 == 0)
				{
					result = 0;
					return result;
				}
			}
			result = 1;
		}
	}
	return result;
}
