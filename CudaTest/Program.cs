using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using OpenCL.Net;

namespace CudaTest
{
    public class Program
    {
        private const int N = 1000000;
        private const int MAX_THREADS = 65535;

        [Cudafy]
        public struct TestClass
        {
            public int A;
            public int B;
            public int C;
        }

        public static void Main()
        {

            Assert.AreEqual(1, isPrime(70657));
            Assert.AreEqual(0, isPrime(70658));
            Assert.AreEqual(0, isPrime(70659));
            Assert.AreEqual(1, isPrime(70663));
            
            CudafyTranslator.GenerateDebug = true;
            CudafyModule cm = CudafyTranslator.Cudafy();

            var gpu = CudafyHost.GetDevice(eGPUType.Cuda, CudafyModes.DeviceId);
            try
            {
                gpu.LoadModule(cm);

                //var test = new TestClass[N];

                //for (var i = 0; i < N; i++)
                //{
                //    test[i] = new TestClass() {A = -i, B = i*i};
                //}

                //var dev_test = gpu.CopyToDevice(test);

                //int[] ints = new int[N];

                //for (var i = 0; i < N; i++)
                //{
                //    ints[i] = i;
                //}

                //var start = DateTime.Now;

                //gpu.Launch(MAX_THREADS, 1).doTheThing(dev_test);
                //TestClass[] testRes = new TestClass[N];
                //gpu.CopyFromDevice(dev_test, testRes);

                //var tt = DateTime.Now - start;

                //start = DateTime.Now;

                //var tt2 = DateTime.Now - start;

                //Console.WriteLine("GPU: {0} ticks, {2}ms, CPU: {1}, {3}ms", tt.Ticks, tt2.Ticks, tt.TotalMilliseconds,
                //                  tt2.TotalMilliseconds);

                //for (var i = 0; i < 10000; i++)
                //{
                    cudaPrimes(gpu);
                //}
            }
            finally
            {
                gpu.FreeAll();
            }
            Console.ReadKey();
        }

        public static void cudaPrimes(GPGPU gpu)
        {
            var timesMax = N/MAX_THREADS;
            var rem = N%MAX_THREADS;
            var index1 = rem > 0 ? timesMax + 1 : timesMax;
            int[][] ints = new int[index1][];
            int[] cpuInts = new int[N];

            for (int i = 0; i < N; i++)
            {
                cpuInts[i] = i;
                var ind1 = i/MAX_THREADS;
                var ind2 = i%MAX_THREADS;
                if (ind2 == 0)
                {
                    ints[ind1] = new int[MAX_THREADS];
                }
                ints[ind1][ind2] = i;
            }

            int[][] totalResults = new int[index1][];
            
            var start = DateTime.Now;
            for (var j = 0; j < index1; j++)
            {
                int[] results = new int[MAX_THREADS];
                var ints_c = gpu.CopyToDevice(ints[j]);

                var results_c = gpu.Allocate<int>(results);
                gpu.Launch(MAX_THREADS, 1).findPrimes(ints_c, results_c);
                gpu.CopyFromDevice(results_c, results);
                totalResults[j] = results;
                gpu.FreeAll();
            }

            var tt = DateTime.Now - start;
            Console.WriteLine("GPU: {0} ticks, {1}ms, found {2} primes", tt.Ticks, tt.TotalMilliseconds,
                              totalResults.Sum(r => r.Sum()));

            start = DateTime.Now;

            Console.WriteLine("Start CPU: " + start.ToLongTimeString());

            // cpu version
            int[] cpuResults = new int[N];
            findPrimesCpu(cpuInts, cpuResults);
            var tt2 = DateTime.Now - start;

            Console.WriteLine("CPU: {0}, {1}ms, found {2} primes", tt2.Ticks, tt2.TotalMilliseconds, cpuResults.Sum());

            //for (int i = 0; i < N; i++)
            //{
            //    Console.WriteLine("Is Prime: {0}, {1}", i, results[i]);
            //}
            
        }

        [Cudafy]
        public static void doTheThing(GThread thread, TestClass[] tests)
        {
            int tid = thread.blockIdx.x;
            if (tid < N)
            {
                var test = tests[tid];
                test.C = test.A + test.B;
            }
        }

        [Cudafy]
        public static void addArrays(GThread thread, int[] a, int[] b, int[] c)
        {
            int tid = thread.blockIdx.x;
            if (tid < N)
                c[tid] = a[tid] + b[tid];
        }

        [Cudafy]
        public static void findPrimes(GThread thread, int[] toCheck, int[] results)
        {
            int tid = thread.blockIdx.x;

            if (tid < N)
            {
                results[tid] = isPrime(toCheck[tid]);
            }
        }

        public static void findPrimesCpu(int[] toCheck, int[] results)
        {
            toCheck.Select(v => isPrime(v)).ToArray().CopyTo(results, 0);
            //for (var i = 0; i < N; i++)
            //{
            //    int b = isPrime(toCheck[i]);
            //    results[i] = b;
            //}
        }

        [Cudafy]
        public static int isPrime(int a)
        {
            if (a == 1 || a == 2)
            {
                return 1;
            }

            int res = a%2;

            if (res == 0)
            {
                return 0;
            }
            int halfWay = a/2;
            for (int i = 3; i <= halfWay; i++)
            {
                if (a%i == 0)
                {
                    return 0;
                }
            }

            return 1;
        }
        
        public static void addArraysCpu(int[] a, int[] b, int[] c)
        {
            for (var tid = 0; tid < N; tid++)
                c[tid] = a[tid] + b[tid];
        }

        private static void OpenCL()
        {
            ErrorCode error;
            Platform[] platformIds = Cl.GetPlatformIDs(out error);
            var p = Cl.GetPlatformInfo(platformIds.First(), PlatformInfo.Name, out error);

            Device[] deviceIds = Cl.GetDeviceIDs(
                (from platform in platformIds
                 select platform).First(), DeviceType.All, out error);
            var deviceName = Cl.GetDeviceInfo(deviceIds.First(), DeviceInfo.Name, out error);
            var deviceLocalMemory = Cl.GetDeviceInfo(deviceIds.First(), DeviceInfo.LocalMemSize, out error);
            var x = Cl.GetDeviceInfo(deviceIds.First(), DeviceInfo.AddressBits, out error);
            var y = Cl.GetDeviceInfo(deviceIds.First(), DeviceInfo.Available, out error);
            var z = Cl.GetDeviceInfo(deviceIds.First(), DeviceInfo.GlobalMemCacheSize, out error);
            var w = Cl.GetDeviceInfo(deviceIds.First(), DeviceInfo.LocalMemType, out error);
        }
    }
}
