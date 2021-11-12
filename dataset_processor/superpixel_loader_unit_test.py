import unittest
from dataset_processor.SuperpixelLoader import SimplicialComplexDataset
from dataset_processor.EdgeFlow import PixelBasedEdgeFlowSC
from dataset_processor.ImageProcessor import ProcessImage
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm

class MyTestCase(unittest.TestCase):

    def test_collate_and_get_return_correct_scData_MNIST(self):
        superpixel_size = 75
        range_size = 10000

        train_data = SimplicialComplexDataset('./data', datasets.MNIST, superpixel_size, PixelBasedEdgeFlowSC, train=True)
        mnist_images = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        PI = ProcessImage(superpixel_size, PixelBasedEdgeFlowSC)

        for i in tqdm(range(range_size)):
            scData_1 = train_data[i]
            scData_2 = PI.image_to_features(mnist_images[i])

            self.assertTrue(scData_1 == scData_2)


    def test_collate_and_get_return_correct_scData_CIFAR10(self):
        superpixel_size = 75
        range_size = 1000

        train_data = SimplicialComplexDataset('./data', datasets.CIFAR10, superpixel_size, PixelBasedEdgeFlowSC, train=True)
        mnist_images = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        PI = ProcessImage(superpixel_size, PixelBasedEdgeFlowSC)

        for i in tqdm(range(range_size)):
            scData_1 = train_data[i]
            scData_2 = PI.image_to_features(mnist_images[i])

            self.assertTrue(scData_1 == scData_2)



if __name__ == '__main__':
    unittest.main()
