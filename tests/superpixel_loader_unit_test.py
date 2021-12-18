import unittest
from dataset_processor.SuperpixelLoader import SimplicialComplexDataset
from dataset_processor.EdgeFlow import PixelBasedEdgeFlow
from dataset_processor.ImageProcessor import ProcessImage
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm
from models.GNN.GNNProcessor import GNNProcessor
from models.SNN_Ebli.SNNEbliProcessor import SNNEbliProcessor
from models.SNN_Bunch.SNNBunchProcessor import SNNBunchProcessor
from models.SAT.model import SAT
from models.SAT.SATProcessor import SATProcessor

class MyTestCase(unittest.TestCase):

    def test_collate_and_get_return_correct_graphObject_MNIST(self):
        superpixel_size = 35
        range_size = 10000
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow
        train = False
        processor_type = GNNProcessor()

        train_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=train)
        mnist_images = datasets.MNIST(root='./data', train=train, download=True, transform=transforms.ToTensor())
        PI = ProcessImage(superpixel_size, PixelBasedEdgeFlow)

        for i in tqdm(range(range_size)):
            graphObject_1 = train_data[i]
            graphObject_2 = processor_type.process(PI.image_to_features(mnist_images[i]))

            self.assertTrue(graphObject_1 == graphObject_2)


    def test_collate_and_get_return_correct_SimplicialObject_Ebli_MNIST(self):
        superpixel_size = 35
        range_size = 10000
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow
        train = False
        processor_type = SNNEbliProcessor()

        train_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=train)
        mnist_images = datasets.MNIST(root='./data', train=train, download=True, transform=transforms.ToTensor())
        PI = ProcessImage(superpixel_size, PixelBasedEdgeFlow)

        for i in tqdm(range(range_size)):
            simplicialObject_1 = train_data[i]
            simplicialObject_2 = processor_type.process(PI.image_to_features(mnist_images[i]))

            self.assertTrue(simplicialObject_1 == simplicialObject_2)


    def test_collate_and_get_return_correct_SimplicialObject_Bunch_MNIST(self):
        superpixel_size = 35
        range_size = 10000
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow
        train = False
        processor_type = SNNBunchProcessor()

        train_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=train)
        mnist_images = datasets.MNIST(root='./data', train=train, download=True, transform=transforms.ToTensor())
        PI = ProcessImage(superpixel_size, PixelBasedEdgeFlow)

        for i in tqdm(range(range_size)):
            simplicialObject_1 = train_data[i]
            simplicialObject_2 = processor_type.process(PI.image_to_features(mnist_images[i]))

            self.assertTrue(simplicialObject_1 == simplicialObject_2)


    def test_collate_and_get_return_correct_SimplicialObject_SAT_MNIST(self):
        superpixel_size = 25
        range_size = 10000
        dataset = datasets.MNIST
        edgeFlow = PixelBasedEdgeFlow
        train = False
        processor_type = SATProcessor()

        train_data = SimplicialComplexDataset('./data', dataset, superpixel_size, edgeFlow, processor_type, train=train)
        mnist_images = datasets.MNIST(root='./data', train=train, download=True, transform=transforms.ToTensor())
        PI = ProcessImage(superpixel_size, PixelBasedEdgeFlow)

        for i in tqdm(range(range_size)):
            simplicialObject_1 = train_data[i]
            simplicialObject_2 = processor_type.process(PI.image_to_features(mnist_images[i]))

            self.assertTrue(simplicialObject_1 == simplicialObject_2)



if __name__ == '__main__':
    unittest.main()
