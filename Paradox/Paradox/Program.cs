using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Paradox
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var inputs = new double[,] { { 0, 1, 1 }, { 0, 0, 0 }, { 1, 0, 1 } };
            var outputs = new[] { 0, 1, 0 };

            var perceptron = new Perceptron(inputs, outputs);
            var weights = perceptron.CalculateWeigths();

            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                var row = inputs.GetRow(i);
                Console.WriteLine(perceptron.Calculate(row, weights));
            }

            Console.ReadLine();
        }
    }

    public class Perceptron
    {
        public double[,] Inputs { get; }
        public int[] Outputs { get; }

        public Perceptron(double[,] inputs, int[] outputs)
        {
            this.Inputs = inputs;
            this.Outputs = outputs;
        }

        public double[] CalculateWeigths()
        {
            var rowSize = this.Inputs.GetLength(0);
            var columnSize = this.Inputs.GetLength(1);

            var random = new Random();
            var weights = Enumerable.Range(0, columnSize).Select(el => random.NextDouble()).ToArray();

            var learningRate = 1.0D;
            var totalError = 1.0D;

            while (totalError > 0.2D)
            {
                totalError = 0;
                for (int i = 0; i < rowSize; i++)
                {
                    double[] row = this.Inputs.GetRow(i);

                    int output = this.Calculate(row, weights);
                    int error = this.Outputs[i] - output;

                    for (var j = 0; j < weights.Length; j++)
                    {
                        weights[j] += learningRate * error * this.Inputs[i, j];
                    }

                    totalError += Math.Abs(error);
                }

            }

            return weights;
        }

        public int Calculate(double[] inputs, double[] weights)
        {
            var sum = inputs.Zip(weights, (input, weigth) => input * weigth).Sum();
            return sum >= 0 ? 1 : 0;
        }
    }

    public static class ArrayExtensions
    {
        public static T[] GetRow<T>(this T[,] matrix, int row)
        {
            var columns = matrix.GetLength(1);
            var array = new T[columns];

            for (int i = 0; i < columns; ++i)
            {
                array[i] = matrix[row, i];
            }

            return array;
        }
    }
}