import unittest
from dataset_processor.SuperpixelLoader import SimplicialComplexDataset, DatasetType
from dataset_processor.EdgeFlow import PixelBasedEdgeFlowSC
import torchvision.transforms as transforms
from torchvision import datasets

class MyTestCase(unittest.TestCase):

    def test_collate_and_get_return_correct_scData(self):
        superpixel_size = 75

        train_data = SimplicialComplexDataset('./data', DatasetType.MNIST, superpixel_size, PixelBasedEdgeFlowSC, train=True)
        mnist_images = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

        for i in range(1):
            scData_1 = train_data[i]
            scData_2, _ = mnist_images[i]

            print(mnist_images[i])
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
