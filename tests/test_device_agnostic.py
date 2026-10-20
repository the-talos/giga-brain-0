"""Device selection tests for the inference entry points. No hardware needed except the XPU case."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

INFERENCE_DIR = Path(__file__).resolve().parents[1] / 'scripts' / 'inference'
sys.path.insert(0, str(INFERENCE_DIR))

import device_utils  # noqa: E402


def fake_host(xpu=False, cuda=False):
    return mock.patch.object(device_utils, 'is_available', side_effect=lambda t: {'xpu': xpu, 'cuda': cuda, 'cpu': True}[t])


class ResolveDeviceTest(unittest.TestCase):
    def test_cpu_only_host(self):
        with fake_host():
            self.assertEqual(device_utils.resolve_device(None), torch.device('cpu'))

    def test_cuda_host_keeps_cuda_default(self):
        with fake_host(cuda=True):
            self.assertEqual(device_utils.resolve_device(None), torch.device('cuda', 0))

    def test_xpu_host_prefers_xpu(self):
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(device_utils.resolve_device(None), torch.device('xpu', 0))

    def test_explicit_device_wins(self):
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(device_utils.resolve_device('cuda:1'), torch.device('cuda', 1))
            self.assertEqual(device_utils.resolve_device('cpu'), torch.device('cpu'))
            self.assertEqual(device_utils.resolve_device('xpu:1'), torch.device('xpu', 1))

    def test_seed_dispatch(self):
        with fake_host(xpu=True, cuda=True), mock.patch.object(torch.cuda, 'manual_seed_all') as cuda_seed, mock.patch.object(
            torch.xpu, 'manual_seed_all'
        ) as xpu_seed:
            device_utils.manual_seed_all(3, 'cuda:0')
            cuda_seed.assert_called_once_with(3)
            xpu_seed.assert_not_called()
            device_utils.manual_seed_all(4, torch.device('xpu', 0))
            xpu_seed.assert_called_once_with(4)

    def test_seed_cpu_is_noop(self):
        with fake_host(xpu=True, cuda=True), mock.patch.object(torch.cuda, 'manual_seed_all') as cuda_seed:
            device_utils.manual_seed_all(3, 'cpu')
            cuda_seed.assert_not_called()


class ServerDeviceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ('tyro', 'giga_models'):
            try:
                importlib.import_module(name)
            except Exception as exc:  # missing optional dependency
                raise unittest.SkipTest(f'{name} is not importable: {exc}')
        spec = importlib.util.spec_from_file_location('gb_server', INFERENCE_DIR / 'inference_agilex_server_unified.py')
        cls.server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.server)

    def run_cli(self, *extra):
        import tyro

        with mock.patch.object(self.server, 'get_policy') as get_policy, mock.patch.object(self.server, 'RobotInferenceServer'):
            tyro.cli(
                self.server.run_server,
                args=['--model-path', 'm', '--pretrained-path', 'p', '--fast-tokenizer-path', 'f', '--norm-stats-path', 'n', *extra],
            )
        return get_policy.call_args.kwargs

    def test_default_device_is_detected_later(self):
        self.assertIsNone(self.run_cli()['device'])

    def test_explicit_device_is_forwarded(self):
        self.assertEqual(self.run_cli('--device', 'xpu:1')['device'], 'xpu:1')

    def test_get_policy_resolves_device_before_loading(self):
        with fake_host(xpu=True), mock.patch.object(self.server, 'resolve_device', wraps=device_utils.resolve_device) as resolve:
            with self.assertRaises(ValueError):  # empty paths stop the call right after device resolution
                self.server.get_policy('', '', '', 0, '', device=None)
        resolve.assert_called_once_with(None)


@unittest.skipUnless(hasattr(torch, 'xpu') and torch.xpu.is_available(), 'needs an XPU')
class XpuSeedTest(unittest.TestCase):
    def test_seed_is_reproducible_on_xpu(self):
        device = torch.device('xpu', 0)
        device_utils.manual_seed_all(7, device)
        first = torch.randn(16, device=device)
        device_utils.manual_seed_all(7, device)
        second = torch.randn(16, device=device)
        self.assertTrue(torch.equal(first, second))
        self.assertGreater(first.std().item(), 0.1)


if __name__ == '__main__':
    unittest.main()
