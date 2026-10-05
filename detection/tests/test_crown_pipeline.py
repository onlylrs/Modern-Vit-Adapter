"""Controller regressions for native NAS access and optional local staging."""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/crown_pipeline.py'
spec = importlib.util.spec_from_file_location('crown_pipeline', SCRIPT)
pipeline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pipeline
spec.loader.exec_module(pipeline)


class NativeNasTests(unittest.TestCase):
    def test_health_check_accepts_native_filesystem_without_findmnt(self):
        with patch.object(pipeline, 'timed_run', return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertTrue(pipeline.mount_healthy())
            self.assertEqual(run.call_args.args[0], ['stat', '-L', str(pipeline.PUBLIC)])

    def test_health_check_handles_unavailable_nas(self):
        with patch.object(pipeline, 'timed_run', side_effect=subprocess.TimeoutExpired('stat', 8)):
            self.assertFalse(pipeline.mount_healthy())

    def test_direct_mode_skips_copy_and_requires_source_checkpoint(self):
        with patch.object(pipeline, 'STAGE_INPUTS', False), \
                patch.object(pipeline, 'checkpoint_source_healthy', return_value=True), \
                patch.object(pipeline.subprocess, 'Popen') as spawn:
            pipeline.stage_pretrained(lambda: False)
            spawn.assert_not_called()
        with patch.object(pipeline, 'STAGE_INPUTS', False), \
                patch.object(pipeline, 'checkpoint_source_healthy', return_value=False):
            with self.assertRaises(pipeline.MountUnavailable):
                pipeline.stage_pretrained(lambda: False)

    def test_staging_remains_optional_and_external_jobs_skip_training(self):
        with tempfile.TemporaryDirectory() as root, patch.object(pipeline, 'LOCAL', Path(root) / 'local'), \
                patch.object(pipeline, 'ARCHIVE', Path(root) / 'nas'):
            job = pipeline.BY_KEY['det_apcdata']
            external = pipeline.BY_KEY['det_cbc']
            with patch.object(pipeline, 'STAGE_INPUTS', False):
                self.assertEqual(pipeline.stage(job), 'train')
                self.assertEqual(pipeline.stage(external), 'eval')
                job.work.mkdir(parents=True)
                (job.work / 'train.done').touch()
                self.assertEqual(pipeline.stage(job), 'eval')
                (job.work / 'eval.done').touch()
                self.assertEqual(pipeline.stage(job), 'sync')
            with patch.object(pipeline, 'STAGE_INPUTS', True):
                self.assertEqual(pipeline.stage(job), 'data')
                job.data.mkdir(parents=True)
                job.data_marker.touch()
                self.assertEqual(pipeline.stage(job), 'sync')

    def test_nas_finalization_preserves_best_and_never_copies_onto_itself(self):
        with tempfile.TemporaryDirectory() as root, \
                patch.object(pipeline, 'ARCHIVE', Path(root) / 'nas'), \
                patch.object(pipeline, 'LOCAL', Path(root) / 'local'):
            job = pipeline.BY_KEY['det_txl_pbc']
            job.work.mkdir(parents=True)
            self.assertEqual(job.work, job.archive)
            best = job.work / 'best_bbox_mAP_epoch_2.pth'
            best.write_bytes(b'best model')
            (job.work / 'epoch_2.pth').write_bytes(b'resume model')
            (job.work / 'latest.pth').symlink_to('epoch_2.pth')
            config = Path(root) / 'config.py'
            config.write_text('model = dict()')
            with patch.object(pipeline.Job, 'config', new_callable=lambda: property(lambda _: config)), \
                    patch.object(pipeline, 'retry_resumable_copy') as copy:
                pipeline.stage_archive(job)
                copy.assert_not_called()
            pipeline.cleanup_checkpoints(job)
            self.assertEqual(best.read_bytes(), b'best model')
            self.assertEqual(list(job.work.glob('*.pth')), [best])
            external = pipeline.BY_KEY['det_cbc']
            external.work.mkdir(parents=True)
            (external.work / best.name).symlink_to(best)
            pipeline.cleanup_checkpoints(external)
            self.assertTrue(best.is_file())
            self.assertEqual(list(external.work.glob('*.pth')), [])
            self.assertFalse((Path(root) / 'local').exists())

    def test_worker_inherits_resolved_paths_and_staging_setting(self):
        with patch.object(pipeline, 'STAGE_INPUTS', False):
            _, _, env = pipeline.command_for(pipeline.BY_KEY['det_apcdata'], 'sync', None)
        self.assertEqual(env['CROWN_PUBLIC_ROOT'], str(pipeline.PUBLIC))
        self.assertEqual(env['CROWN_ARCHIVE_ROOT'], str(pipeline.ARCHIVE))
        self.assertEqual(env['CROWN_STAGE_INPUTS'], '0')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')


if __name__ == '__main__':
    unittest.main()
