"""Unit tests for SnapshotManager class."""

import tempfile
import shutil
from pathlib import Path
from unittest import TestCase

from merkle.merkle_dag import MerkleDAG
from merkle.snapshot_manager import SnapshotManager


class TestSnapshotManager(TestCase):
    """Test SnapshotManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = str(Path(tempfile.mkdtemp()).resolve())
        self.test_path = Path(self.temp_dir) / 'project'
        self.test_path.mkdir()

        self.storage_dir = Path(self.temp_dir) / 'snapshots'
        self.manager = SnapshotManager(self.storage_dir)

        # Create test files
        (self.test_path / 'test.py').write_text('print("test")')

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_project_id_generation(self):
        """Test project ID generation."""
        id1 = self.manager.get_project_id('/path/to/project')
        id2 = self.manager.get_project_id('/path/to/project')
        id3 = self.manager.get_project_id('/different/path')

        # Same path should produce same ID
        assert id1 == id2

        # Different path should produce different ID
        assert id1 != id3

    def test_save_and_load_snapshot(self):
        """Test saving and loading snapshots."""
        # Create DAG
        dag = MerkleDAG(str(self.test_path))
        dag.build()

        # Save snapshot
        self.manager.save_snapshot(dag, {'test': 'metadata'})

        # Load snapshot
        loaded_dag = self.manager.load_snapshot(str(self.test_path))

        assert loaded_dag is not None
        assert loaded_dag.get_root_hash() == dag.get_root_hash()
        assert loaded_dag.get_all_files() == dag.get_all_files()

    def test_metadata_handling(self):
        """Test metadata save and load."""
        dag = MerkleDAG(str(self.test_path))
        dag.build()

        # Save with metadata
        custom_metadata = {'version': '1.0', 'author': 'test'}
        self.manager.save_snapshot(dag, custom_metadata)

        # Load metadata
        metadata = self.manager.load_metadata(str(self.test_path))

        assert metadata is not None
        assert metadata['version'] == '1.0'
        assert metadata['author'] == 'test'
        assert metadata['project_path'] == str(self.test_path)
        assert metadata['file_count'] == 1

    def test_snapshot_existence_check(self):
        """Test checking if snapshot exists."""
        assert not self.manager.has_snapshot(str(self.test_path))

        dag = MerkleDAG(str(self.test_path))
        dag.build()
        self.manager.save_snapshot(dag)

        assert self.manager.has_snapshot(str(self.test_path))

    def test_list_snapshots(self):
        """Test listing all snapshots."""
        import time

        # Create multiple project snapshots
        for i in range(3):
            project_path = self.test_path.parent / f'project{i}'
            project_path.mkdir()
            (project_path / 'file.txt').write_text(f'content{i}')

            dag = MerkleDAG(str(project_path))
            dag.build()
            self.manager.save_snapshot(dag)

            time.sleep(0.1)  # Ensure different timestamps

        snapshots = self.manager.list_snapshots()

        assert len(snapshots) == 3
        # Should be sorted by timestamp (most recent first)
        assert 'project2' in snapshots[0]['project_path']
