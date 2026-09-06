"""Offline selection, download, and partition integration checks."""
import contextlib
import gzip
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import dataset_catalog as catalog
import fetch_data as fetch
import generate_partitions as partitions


class OpenDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original_output = partitions.OUTPUT_DIR
        self.addCleanup(partitions.configure_output, self.original_output)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def test_selection_and_unknown_license(self):
        self.assertEqual({d.id for d in catalog.select_datasets(True)},
                         {'mango', 'melamine', 'eggs', 'wheat', 'ossl'})
        self.assertEqual(len(catalog.select_datasets()), 9)
        self.assertEqual(sum(len(d.files) for d in catalog.select_datasets()), 10)
        unknown = catalog.Dataset('new', (), 'Unknown', '', '')
        with patch.object(catalog, 'DATASETS', (*catalog.DATASETS, unknown)):
            self.assertNotIn(unknown, catalog.select_datasets(True))
        with self.assertRaises(ValueError):
            catalog.select_partitions('corn', True)
        self.assertEqual(catalog.select_partitions(open_only=True),
                         ('melamine', 'eggs', 'soil_nir', 'soil_mir', 'mango', 'wheat'))

    def test_dry_run_has_no_network_or_files(self):
        target = self.root / 'downloads'
        with patch.object(fetch, 'DOWNLOAD_DIR', str(target)), patch.object(fetch.requests, 'get') as get:
            self.assertEqual(fetch.main(['--open-only', '--dry-run']), 0)
            get.assert_not_called()
            self.assertFalse(target.exists())
        with self.assertRaises(SystemExit) as error:
            fetch.main(['--unknown-option'])
        self.assertEqual(error.exception.code, 2)

    def test_open_download_never_requests_excluded_urls(self):
        class Response:
            status_code = 200
            headers = {}
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def raise_for_status(self):
                pass
            def iter_content(self, chunk_size):
                yield self.payload
        def respond(url, **kwargs):
            response = Response()
            response.payload = gzip.compress(b'a,b\n1,2\n') if url.endswith('.gz') else b'fixture'
            return response
        with patch.object(fetch, 'DOWNLOAD_DIR', str(self.root)), patch.object(fetch.requests, 'get', side_effect=respond) as get:
            fetch.main(['--open-only'])
        expected = [url for d in catalog.select_datasets(True) for url, _ in d.files]
        self.assertEqual([call.args[0] for call in get.call_args_list], expected)
        self.assertFalse(any('eigenvector' in url for url in expected))
        self.assertEqual((self.root / 'ossl_all_L0_v1.2.csv').read_bytes(), b'a,b\n1,2\n')
        self.assertFalse(list(self.root.glob('*.gz')))

    def test_download_fallback_uses_same_publisher_file(self):
        from unittest.mock import MagicMock
        bad = MagicMock()
        bad.__enter__.return_value = bad
        bad.raise_for_status.side_effect = fetch.requests.HTTPError('403 Forbidden')
        good = MagicMock()
        good.__enter__.return_value = good
        good.status_code = 200
        good.headers = {'content-length': '7', 'content-type': 'text/csv'}
        good.iter_content.return_value = [b'fixture']
        output = self.root / 'data.csv'
        with patch.object(fetch.requests, 'get', side_effect=[bad, good]) as get:
            fetch.download_file('https://primary/file', str(output), ('https://publisher/file',))
        self.assertEqual(output.read_bytes(), b'fixture')
        self.assertEqual([call.args[0] for call in get.call_args_list],
                         ['https://primary/file', 'https://publisher/file'])
        self.assertTrue(all(call.kwargs['timeout'] == (30, 120) for call in get.call_args_list))
        self.assertFalse(list(self.root.glob('*.part')))

    def test_download_preserves_existing_permissions(self):
        from unittest.mock import MagicMock
        output = self.root / 'data.csv'
        output.write_bytes(b'old')
        output.chmod(0o640)
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.headers = {'content-length': '3', 'content-type': 'text/csv'}
        response.iter_content.return_value = [b'new']
        with patch.object(fetch.requests, 'get', return_value=response):
            fetch.download_file('https://publisher/file', str(output))
        self.assertEqual(output.read_bytes(), b'new')
        self.assertEqual(os.stat(output).st_mode & 0o777, 0o640)

    def test_invalid_download_preserves_existing_file(self):
        from unittest.mock import MagicMock
        cases = [(202, 'text/html', '0', []), (200, 'text/html', '7', [b'fixture']),
                 (200, 'text/csv', '0', []), (200, 'text/csv', '99', [b'short'])]
        output = self.root / 'data.csv'
        output.write_bytes(b'previous complete data')
        for code, content_type, length, chunks in cases:
            response = MagicMock()
            response.__enter__.return_value = response
            response.status_code = code
            response.headers = {'content-type': content_type, 'content-length': length}
            response.iter_content.return_value = chunks
            with self.subTest(code=code, length=length), patch.object(fetch.requests, 'get', return_value=response):
                with self.assertRaises(fetch.requests.RequestException):
                    fetch.download_file('https://publisher/file', str(output))
            self.assertEqual(output.read_bytes(), b'previous complete data')
            self.assertFalse(list(self.root.glob('*.part')))

    def test_default_download_still_selects_every_file(self):
        with patch.object(fetch, 'DOWNLOAD_DIR', str(self.root)), patch.object(fetch, 'download_file') as download, patch.object(fetch, 'unpack_archive'), patch.object(fetch, 'decompress_gzip', return_value='soil.csv'), patch.object(fetch.os, 'remove'):
            fetch.main([])
        self.assertEqual([call.args[0] for call in download.call_args_list],
                         [url for d in catalog.DATASETS for url, _ in d.files])

    def prepare_inputs(self):
        raw = self.root / 'raw'
        raw.mkdir()
        indices = self.root / 'indices'
        trip = indices / 'TRIP'
        (trip / 'Eggs').mkdir(parents=True)
        pd.DataFrame({'sample': [1, 2, 3], 'storage_days': [0, 1, 2],
                      'Spectra_740': [0.1, 0.2, 0.3]}).to_csv(raw / 'eggs.csv', index=False)
        pd.DataFrame({'set': ['support', 'query', 'support']}).to_csv(trip / 'Eggs' / 'split.csv')
        pd.DataFrame({'task': ['Eggs', 'Corn_Oil', 'Wheat'],
                      'split': ['train', 'train', 'test']}).to_csv(trip / 'splits.csv')
        self.stack.enter_context(patch.object(partitions, 'DATA_ORIG', raw))
        self.stack.enter_context(patch.object(partitions, 'DATA_BASE', indices))
        self.stack.enter_context(patch.object(partitions, 'TRIP_INDICES', trip))
        return raw, trip

    def test_real_eggs_processor_and_filtered_index(self):
        self.prepare_inputs()
        out = self.root / 'open'
        self.assertEqual(partitions.main(['--open-only', '--dataset', 'eggs', '--output-dir', str(out)]), 0)
        index = pd.read_csv(out / 'TRIP' / 'splits.csv', index_col=0)
        self.assertEqual(index.to_dict('list'), {'task': ['Eggs'], 'split': ['train']})
        support = pd.read_csv(out / 'TRIP' / 'Eggs' / 'X_supp.csv', index_col=0)
        self.assertEqual(support.index.tolist(), [0, 2])
        self.assertEqual(support['740'].tolist(), [0.1, 0.3])
        self.assertEqual({p.name for p in (out / 'TRIP').iterdir()}, {'Eggs', 'splits.csv'})

    def test_open_dispatch_skips_excluded_processors(self):
        self.prepare_inputs()
        mocks = {name: self.stack.enter_context(patch.object(partitions, 'process_' + name))
                 for name in catalog.PARTITION_SOURCES}
        self.stack.enter_context(patch.object(partitions, 'write_trip_splits'))
        with patch.object(partitions, 'BASE_DIR', self.root):
            self.assertEqual(partitions.main(['--open-only']), 0)
        self.assertEqual(partitions.OUTPUT_DIR, (self.root / 'data_open').resolve())
        for name, mock in mocks.items():
            self.assertEqual(mock.call_count, int(name in catalog.select_partitions(open_only=True)))

    def test_real_trip_catalog_keeps_all_six_open_tasks(self):
        original = pd.read_csv(partitions.TRIP_INDICES / 'splits.csv', index_col=0)
        expected = original[original['task'].isin([
            'Melamine_R562', 'Melamine_R861', 'Melamine_R568',
            'Melamine_R862', 'Eggs', 'Wheat'])]
        self.assertEqual(len(expected), 6)
        partitions.configure_output(self.root / 'out')
        for task in expected['task']:
            directory = partitions.OUTPUT_TRIP / task
            directory.mkdir(parents=True)
            for name in ('X_supp.csv', 'y_supp.csv', 'X_query.csv', 'y_query.csv'):
                (directory / name).write_text('fixture')
        partitions.write_trip_splits(catalog.select_partitions(open_only=True))
        result = pd.read_csv(partitions.OUTPUT_TRIP / 'splits.csv', index_col=0)
        pd.testing.assert_frame_equal(result, expected)
        self.assertEqual(result.groupby('split').size().to_dict(),
                         {'train': 3, 'val': 1, 'test': 2})

    def test_conflicts_rejected_before_writes(self):
        raw, _ = self.prepare_inputs()
        out = self.root / 'existing'
        out.mkdir()
        marker = out / 'keep.txt'
        marker.write_text('keep')
        cases = [ ['--open-only', '--dataset', 'diesel'],
                  ['--open-only', '--cleanup'],
                  ['--dataset', 'eggs', '--cleanup'],
                  ['--open-only', '--output-dir', str(out)],
                  ['--output-dir', str(raw)],
                  ['--output-dir', str(raw / 'nested')],
                  ['--output-dir', str(self.root)] ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                partitions.main(args)
            self.assertEqual(error.exception.code, 2)
        self.assertEqual(marker.read_text(), 'keep')
        self.assertTrue(raw.exists())

    def test_missing_input_fails_and_does_not_clean_up(self):
        raw, _ = self.prepare_inputs()
        (raw / 'eggs.csv').unlink()
        self.assertEqual(partitions.main(['--open-only', '--dataset', 'eggs', '--output-dir', str(self.root / 'out')]), 1)
        self.assertTrue(raw.exists())
        self.assertFalse((self.root / 'out' / 'TRIP' / 'splits.csv').exists())
        with self.assertRaises(FileNotFoundError):
            partitions.process_wheat()

    def test_soil_processor_uses_only_indexed_tasks(self):
        indices = self.root / 'indices'
        included = indices / 'Soil_Included-pH'
        excluded = indices / 'Soil_Excluded-pH'
        included.mkdir(parents=True)
        excluded.mkdir()
        pd.DataFrame({'task': ['Soil_Included-pH'], 'split': ['train']}).to_csv(
            indices / 'splits.csv')
        split = pd.DataFrame({'set': ['support', 'query']})
        split.to_csv(included / 'split.csv')
        split.to_csv(excluded / 'split.csv')
        soil = self.root / 'soil.csv'
        pd.DataFrame({'scan_1': [0.1, 0.2], 'pH': [6.0, 7.0]}).to_csv(soil)
        output = self.root / 'output'
        partitions.process_soil_dataset('NIR', indices, output, soil)
        self.assertTrue((output / 'Soil_Included-pH').is_dir())
        self.assertFalse((output / 'Soil_Excluded-pH').exists())
        result = pd.read_csv(output / 'splits.csv', index_col=0)
        self.assertEqual(result['task'].tolist(), ['Soil_Included-pH'])

    def test_soil_processor_rejects_missing_indexed_directory(self):
        indices = self.root / 'indices'
        indices.mkdir()
        pd.DataFrame({'task': ['Soil_Missing-pH'], 'split': ['train']}).to_csv(
            indices / 'splits.csv')
        soil = self.root / 'soil.csv'
        pd.DataFrame({'scan_1': [0.1], 'pH': [6.0]}).to_csv(soil)
        with self.assertRaisesRegex(FileNotFoundError, 'Soil_Missing-pH'):
            partitions.process_soil_dataset('NIR', indices, self.root / 'output', soil)

    def test_trip_index_preserves_splits_and_handles_spaces(self):
        _, trip = self.prepare_inputs()
        partitions.configure_output(self.root / 'out')
        for name in ['Eggs', 'Corn_Oil   ']:
            directory = partitions.OUTPUT_TRIP / name
            directory.mkdir(parents=True)
            for filename in ['X_supp.csv', 'y_supp.csv', 'X_query.csv', 'y_query.csv']:
                (directory / filename).write_text('fixture')
        partitions.write_trip_splits(('eggs', 'corn'))
        original = pd.read_csv(trip / 'splits.csv', index_col=0)
        result = pd.read_csv(partitions.OUTPUT_TRIP / 'splits.csv', index_col=0)
        pd.testing.assert_frame_equal(result, original.iloc[:2])
        with self.assertRaises(FileNotFoundError):
            partitions.write_trip_splits(('wheat',))


if __name__ == '__main__':
    unittest.main()
