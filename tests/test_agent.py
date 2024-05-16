import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from ..agent import ModelCatalog, RLlibTorchSavedModelAgent


class TestRLlibTorchSavedModelAgent(unittest.TestCase):
    def setUp(self):
        self.mock_path_to_model = Path("/path/to/model")
        self.mock_observation_space = Mock()
        self.agent = RLlibTorchSavedModelAgent(
            self.mock_path_to_model, self.mock_observation_space
        )

    def test_init(self):
        self.assertEqual(
            self.agent._prep,
            ModelCatalog.get_preprocessor_for_space(self.mock_observation_space),
        )
        self.assertEqual(self.agent._model, torch.load(str(self.mock_path_to_model)))
        self.assertTrue(self.agent._model.eval.called)

    @patch("torch.tensor")
    def test_act(self, mock_torch_tensor):
        mock_obs = Mock()
        mock_torch_tensor.return_value.float.return_value = np.array([0.5])
        mock_action = (
            self.agent._model.return_value.detach.return_value.numpy.return_value
        ) = np.array([0.5])
        action = self.agent.act(mock_obs)
        mock_torch_tensor.assert_called_once_with(self.agent._prep.transform(mock_obs))
        self.assertEqual(action, mock_action)


if __name__ == "__main__":
    unittest.main()
