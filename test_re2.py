import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

class TestTorchVisionVideoClassifier(unittest.TestCase):

    @patch('action_re2.select_device')
    @patch('action_re2.TorchVisionVideoClassifier.model_name_to_model_and_weights')
    def test_initialization_with_valid_model(self, mock_model_weights, mock_select_device):
        mock_model = MagicMock()
        mock_weights = MagicMock()
        mock_model_weights.__getitem__.return_value = (mock_model, mock_weights)
        mock_select_device.return_value = 'cpu'

        classifier = TorchVisionVideoClassifier('s3d')

        self.assertEqual(classifier.device, 'cpu')
        self.assertEqual(classifier.weights, mock_weights)
        mock_model.assert_called_once_with(weights=mock_weights)

    def test_preprocess_crops_for_video_cls(self):
        classifier = TorchVisionVideoClassifier('s3d')
        crops = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(8)]
        processed_crops = classifier.preprocess_crops_for_video_cls(crops)

        self.assertEqual(processed_crops.shape, (1, 3, 8, 224, 224))

    @patch('action_re2.TorchVisionVideoClassifier.model')
    def test_call(self, mock_model):
        classifier = TorchVisionVideoClassifier('s3d')
        sequences = torch.rand(1, 3, 8, 224, 224)
        classifier(sequences)

        mock_model.assert_called_once_with(sequences)

    def test_postprocess(self):
        classifier = TorchVisionVideoClassifier('s3d')
        outputs = torch.rand(8, 400)
        pred_labels, pred_confs = classifier.postprocess(outputs)

        self.assertEqual(len(pred_labels), 8)
        self.assertEqual(len(pred_confs), 8)

class TestHuggingFaceVideoClassifier(unittest.TestCase):

    @patch('action_re2.select_device')
    @patch('action_re2.AutoProcessor.from_pretrained')
    @patch('action_re2.AutoModel.from_pretrained')
    def test_initialization_with_valid_model(self, mock_model, mock_processor, mock_select_device):
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        mock_select_device.return_value = 'cpu'

        classifier = HuggingFaceVideoClassifier(['label1', 'label2'])

        self.assertEqual(classifier.device, 'cpu')
        self.assertEqual(classifier.processor, mock_processor_instance)
        self.assertEqual(classifier.model, mock_model_instance)

    def test_preprocess_crops_for_video_cls(self):
        classifier = HuggingFaceVideoClassifier(['label1', 'label2'])
        crops = [np.random.rand(224, 224, 3).astype(np.float32) for _ in range(8)]
        processed_crops = classifier.preprocess_crops_for_video_cls(crops)

        self.assertEqual(processed_crops.shape, (1, 8, 3, 224, 224))

    @patch('action_re2.HuggingFaceVideoClassifier.model')
    @patch('action_re2.HuggingFaceVideoClassifier.processor')
    def test_call(self, mock_processor, mock_model):
        classifier = HuggingFaceVideoClassifier(['label1', 'label2'])
        sequences = torch.rand(1, 8, 3, 224, 224)
        input_ids = torch.randint(0, 100, (1, 2))
        mock_processor.return_value = {'input_ids': input_ids}
        classifier(sequences)

        mock_model.assert_called_once_with(pixel_values=sequences, input_ids=input_ids)

    def test_postprocess(self):
        classifier = HuggingFaceVideoClassifier(['label1', 'label2'])
        outputs = torch.rand(1, 2)
        pred_labels, pred_confs = classifier.postprocess(outputs)

        self.assertEqual(len(pred_labels), 1)
        self.assertEqual(len(pred_confs), 1)
        self.assertEqual(len(pred_labels[0]), 2)
        self.assertEqual(len(pred_confs[0]), 2)

if __name__ == '__main__':
    unittest.main()