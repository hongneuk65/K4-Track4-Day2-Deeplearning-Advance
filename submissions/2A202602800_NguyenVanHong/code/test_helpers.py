"""CPU checks for helpers; synthetic inputs, no dataset or test evaluation."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import unittest
import tempfile
from unittest.mock import patch
import json
import pandas as pd
import torch
import numpy as np
import timm
import losses
import inference
import benchmark
import train
import model


class HelperTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        torch.set_num_threads(2)

    def test_losses_reduce_to_ce(self):
        z = torch.randn(8, 9); y = torch.randint(9, (8,))
        ce = torch.nn.functional.cross_entropy(z, y)
        torch.testing.assert_close(losses.FocalLoss(0)(z, y), ce)
        torch.testing.assert_close(losses.LabelSmoothingCE(0)(z, y), ce)

    def test_cutmix_area_and_loss(self):
        x = torch.stack([torch.full((3, 32, 32), float(i)) for i in range(8)])
        mixed, (a, b, lam) = losses.mix_batch(x, torch.arange(8))
        for i in range(8):
            if a[i] != b[i]:
                self.assertAlmostEqual((mixed[i, 0] != x[i, 0]).float().mean().item(), 1-lam)
        z = torch.randn(8, 9); criterion = torch.nn.CrossEntropyLoss()
        torch.testing.assert_close(losses.mixed_loss(criterion,z,(a,b,lam)),
                                   lam*criterion(z,a)+(1-lam)*criterion(z,b))

    def test_bn_fusion_sequential_and_resnet(self):
        modules = [torch.nn.Sequential(torch.nn.Conv2d(3,4,3,padding=1),torch.nn.BatchNorm2d(4)),
                   timm.create_model('resnet18',pretrained=False,num_classes=9)]
        x = torch.randn(2,3,32,32)
        for original in modules:
            original.eval(); fused = inference.fuse_conv_bn(original)
            with torch.inference_mode():
                torch.testing.assert_close(original(x),fused(x),atol=1e-5,rtol=1e-5)
            self.assertTrue(any(isinstance(m,torch.nn.BatchNorm2d) for m in original.modules()))

    def test_temperature_and_views(self):
        z = np.random.randn(12,9)
        p = inference.apply_temperature(z,.6)
        np.testing.assert_allclose(p, inference.apply_temperature(z, T=.6))
        np.testing.assert_allclose(p, inference.apply_temperature(z, temperature=.6))
        np.testing.assert_allclose(p.sum(1),1)
        np.testing.assert_array_equal(p.argmax(1),z.argmax(1))
        x = torch.randn(2,3,32,32)
        self.assertEqual([tuple(v.shape) for v in inference.views_multiscale(x,[16,24])],
                         [(2,3,16,16),(2,3,24,24)])
        self.assertEqual(len(inference.views_multicrop(x,24)),5)

    def test_ema_parameters_and_buffers(self):
        m = torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3))
        ema = train.EMA(m,.5)
        before = [p.clone() for p in ema.model.parameters()]
        with torch.no_grad():
            for p in m.parameters(): p.add_(2)
            m[1].running_mean.fill_(3)
            m[1].num_batches_tracked.fill_(7)
        ema.update(m)
        for old,new in zip(before,ema.model.parameters()): torch.testing.assert_close(new,old+1)
        torch.testing.assert_close(ema.model[1].running_mean,m[1].running_mean)
        self.assertEqual(ema.model[1].num_batches_tracked.item(),7)

    def test_cli_and_optimizer_groups(self):
        self.assertEqual(train.parse_overrides(['ema_decay=.9']), {'ema_decay': .9})
        self.assertIsNone(train.parse_overrides(['ema_decay=none'])['ema_decay'])
        self.assertEqual(train.parse_overrides(['seed=2','amp=false','mix=none','class_weight_beta=.9']),
                         dict(seed=2,amp=False,mix=None,class_weight_beta=.9))
        for args in [['unknown=1'],['amp=yes'],['seed=none']]:
            with self.assertRaises(ValueError): train.parse_overrides(args)
        m=timm.create_model('resnet18',pretrained=False,num_classes=9)
        groups=model.param_groups(m,1e-4,1e-3,.05)
        ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in m.parameters() if p.requires_grad})
        for g in groups:
            for p in g['params']:
                if p.ndim<=1: self.assertEqual(g['weight_decay'],0)

    def test_cpu_latency_helpers(self):
        m=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*8*8,9)).eval()
        for result in [benchmark.latency_report(m,1,8,device='cpu',iters=50),
                       benchmark.tta_latency(m,2,batch_size=1,img_size=8,device='cpu',iters=50)]:
            self.assertEqual(result['n'],50)
            self.assertLessEqual(result['p50'],result['p95'])
            self.assertLessEqual(result['p95'],result['p99'])

    def test_run_ema_checkpoint_and_resume(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(3, 9)
                self.pretrained_cfg = {'tag': 'synthetic'}
            def forward(self, x):
                return self.head(x)
            def get_classifier(self):
                return self.head

        frames = pd.DataFrame({'Filename': [f'{i}.jpg' for i in range(9)],
                               'Label': list(range(9))})
        loader = [(torch.randn(9, 3), torch.arange(9), frames.Filename.tolist())]
        observed = []
        real_evaluate = train.evaluate
        def capture(m, *args):
            observed.append({k: v.clone() for k, v in m.state_dict().items()})
            return real_evaluate(m, *args)

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(train.torch.cuda, 'is_available', return_value=False), \
             patch.object(train.dataset, 'load_split', return_value=(frames, frames, frames)), \
             patch.object(train.dataset, 'check_split'), \
             patch.object(train.dataset, 'make_loader', return_value=loader), \
             patch.object(train.model_utils, 'build_model', side_effect=lambda *a, **k: Tiny()), \
             patch.object(train.model_utils, 'count_gmacs', return_value=0.0), \
             patch.object(train, 'evaluate', side_effect=capture):
            cfg = train.Config(epochs=2, ema_decay=.5, amp=False,
                               out_dir=directory+'/runs', pred_dir=directory+'/predictions',
                               curves_dir=directory+'/curves')
            first = train.run(cfg)
            best = torch.load(train.run_dir(cfg)/'best.pt', weights_only=False)
            last = torch.load(train.run_dir(cfg)/'last.pt', weights_only=False)
            for k, v in best['model'].items():
                torch.testing.assert_close(v, observed[first['best_epoch']-1][k])
            self.assertTrue(any(not torch.equal(last['model'][k], last['ema'][k])
                                for k in last['model']))
            second = train.run(cfg)
            self.assertAlmostEqual(first['macro_f1_val'], second['macro_f1_val'])
            self.assertFalse(list(Path(directory+'/predictions').glob('*test.csv')))

    def test_resume_legacy_config_without_ema(self):
        # Default None preserves the old submitted experiment config fields.
        from dataclasses import asdict
        previous = asdict(train.Config())
        del previous['ema_decay']
        previous.setdefault('ema_decay', None)
        self.assertEqual(previous, asdict(train.Config()))


if __name__ == '__main__':
    unittest.main()
