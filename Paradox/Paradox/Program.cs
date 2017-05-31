using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using System.Drawing;
using System.IO;
using Emgu.CV.Util;
using Emgu.CV.Features2D;

// Author: Alesson Ricardo Bernardo
// Author: Gabriel Takashi Katakura
// Author: Ivan Manoel da Silva Filho
namespace Paradox
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var perceptron = TrainingPerceptron();
            var directoryName = @"C:\Users\Gabriel\Downloads\dataset_geral";

            foreach (var fileName in Directory.GetFiles(directoryName))
            {
                var originalImg = CvInvoke.Imread(fileName, LoadImageType.AnyColor);

                var imageWithoutBorder = ChangeColor(originalImg.Clone(), new Rgb(0D, 0D, 0D), new Rgb(255D, 255D, 255D));

                var imageGray = imageWithoutBorder.Clone();
                CvInvoke.CvtColor(imageWithoutBorder, imageGray, ColorConversion.Bgr2Gray);

                var nucleusImageBinary = new Mat();
                CvInvoke.Threshold(imageGray, nucleusImageBinary, 100, 255, ThresholdType.Binary);

                var nucleusImageBinaryInv = new Mat();
                CvInvoke.Threshold(nucleusImageBinary, nucleusImageBinaryInv, 254, 255, ThresholdType.BinaryInv);

                var imgOut = nucleusImageBinaryInv.Clone();
                var element = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(3, 3), new Point(2, 2));
                CvInvoke.MorphologyEx(imgOut, imgOut, MorphOp.Erode, element, default(Point), 1, BorderType.Constant, new MCvScalar(0, 0));
                CvInvoke.MorphologyEx(imgOut, imgOut, MorphOp.Dilate, element, default(Point), 1, BorderType.Constant, new MCvScalar(0, 0));

                var nucleuses = new VectorOfVectorOfPoint();
                CvInvoke.FindContours(imgOut, nucleuses, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                double nucleusArea = 0D;

                for (var i = 0; i < nucleuses.Size; i++)
                {
                    nucleusArea += CvInvoke.ContourArea(nucleuses[i]);
                }

                var cytoplasmImageInv = new Mat();
                CvInvoke.CvtColor(imageWithoutBorder, cytoplasmImageInv, ColorConversion.Bgr2Gray);
                CvInvoke.Threshold(cytoplasmImageInv, cytoplasmImageInv, 160, 255, ThresholdType.BinaryInv);

                var cytoplasmContours = new VectorOfVectorOfPoint();
                CvInvoke.FindContours(cytoplasmImageInv, cytoplasmContours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);
                var cytoplasmArea = 0D;

                for (var i = 0; i < cytoplasmContours.Size; i++)
                {
                    cytoplasmArea += CvInvoke.ContourArea(cytoplasmContours[i]);
                }

                var nucleusPercentFromCytoplasm = nucleusArea * 100 / cytoplasmArea;

                var isCircular = IsCircular(nucleuses, imgOut);

                var characteristics = new double[] {
                    isCircular ? 1 : 0,
                    nucleuses.Size > 1 ? 1 : 0,
                    nucleusPercentFromCytoplasm > 70D ? 1 : 0
                };

                var category = perceptron.Calculate(characteristics) == 1 ? "Neutrófilo" : "Linfócito";

                Console.WriteLine($"{Path.GetFileName(fileName)} = {category}");
                Console.WriteLine($"- Circular = {(isCircular ? "Sim" : "Não")}");
                Console.WriteLine($"- Núcleos = {nucleuses.Size}");
                Console.WriteLine($"- Área ocupada = {nucleusPercentFromCytoplasm}");
                Console.WriteLine();
            }

            Console.ReadLine();
        }

        public static UMat ChangeColor(Mat mat, Rgb oldColor, Rgb newColor)
        {
            var image = mat.ToImage<Rgb, byte>();

            for (int i = 0; i < image.Rows; i++)
            {
                for (int j = 0; j < image.Cols; j++)
                {
                    var currentColor = image[i, j];

                    if (currentColor.Equals(oldColor))
                    {
                        image[i, j] = newColor;
                    }
                }
            }

            return image.ToUMat();
        }

        public static bool IsCircular(VectorOfVectorOfPoint nucleuses, Mat image)
        {
            if (nucleuses.Size > 1)
            {
                return false;
            }

            var nucleus = new VectorOfVectorOfPoint(nucleuses[0]);
            var nucleusImage = image.Clone();

            CvInvoke.Threshold(nucleusImage, nucleusImage, 254, 255, ThresholdType.BinaryInv);
            CvInvoke.DrawContours(nucleusImage, nucleus, -1, new MCvScalar(0, 0, 255), -1);

            var circles = CvInvoke.HoughCircles(nucleusImage, HoughType.Gradient, 10, 100, 100, 64);

            return circles.Length > 0;
        }

        public static Perceptron TrainingPerceptron()
        {
            var inputs = new double[,] { { 1, 0, 1 }, { 0, 1, 0 } };
            var outputs = new[] { 0, 1 };

            return new PerceptronTraining(inputs, outputs).Trained();
        }
    }

    public class Perceptron
    {
        public double[] Weights { get; }

        public Perceptron(double[] weights)
        {
            this.Weights = weights;
        }

        public int Calculate(double[] inputs)
        {
            var sum = inputs.Zip(this.Weights, (input, weigth) => input * weigth).Sum();
            return sum >= 0 ? 1 : 0;
        }
    }

    public class PerceptronTraining
    {
        public double[,] Inputs { get; }
        public int[] Outputs { get; }

        public PerceptronTraining(double[,] inputs, int[] outputs)
        {
            this.Inputs = inputs;
            this.Outputs = outputs;
        }

        public Perceptron Trained()
        {
            return new Perceptron(this.Weigths());
        }

        public double[] Weigths()
        {
            var rowSize = this.Inputs.GetLength(0);
            var columnSize = this.Inputs.GetLength(1);

            var random = new Random();
            var weights = Enumerable.Range(0, columnSize).Select(el => random.NextDouble()).ToArray();

            var learningRate = 1.0D;
            var totalError = 1.0D;

            while (totalError > 0D)
            {
                totalError = 0;
                for (int i = 0; i < rowSize; i++)
                {
                    double[] row = this.Inputs.GetRow(i);

                    int output = new Perceptron(weights).Calculate(row);
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