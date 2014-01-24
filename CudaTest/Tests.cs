using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CudaTest
{
    [TestClass]
    public class Tests
    {
        [TestMethod]
        public void Test70657()
        {
            Assert.AreEqual(1, Program.isPrime(70657));
        }
    }
}
