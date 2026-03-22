"""Unit tests for ChangeDetector class."""

import tempfile
import shutil
from pathlib import Path
from unittest import TestCase

from merkle.merkle_dag import MerkleDAG
from merkle.snapshot_manager import SnapshotManager
from merkle.change_detector import ChangeDetector, FileChanges


class TestChangeDetector(TestCase):
    """Test ChangeDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = str(Path(tempfile.mkdtemp()).resolve())
        self.test_path = Path(self.temp_dir)

        self.storage_dir = Path(self.temp_dir) / 'snapshots'
        self.snapshot_manager = SnapshotManager(self.storage_dir)
        self.detector = ChangeDetector(self.snapshot_manager)

        self.create_initial_files()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_initial_files(self):
        """Create initial file structure."""
        (self.test_path / 'src').mkdir()
        (self.test_path / 'unchanged.py').write_text('# unchanged')
        (self.test_path / 'to_modify.py').write_text('# original')
        (self.test_path / 'to_remove.py').write_text('# remove me')
        (self.test_path / 'src' / 'module.py').write_text('# module')

    def test_detect_changes_between_dags(self):
        """Test detecting changes between two DAGs."""
        # Create initial DAG
        dag1 = MerkleDAG(str(self.test_path))
        dag1.build()

        # Modify files
        (self.test_path / 'to_modify.py').write_text('# modified')
        (self.test_path / 'to_remove.py').unlink()
        (self.test_path / 'added.py').write_text('# new file')

        # Create new DAG
        dag2 = MerkleDAG(str(self.test_path))
        dag2.build()

        # Detect changes
        changes = self.detector.detect_changes(dag1, dag2)

        assert 'added.py' in changes.added
        assert 'to_remove.py' in changes.removed
        assert 'to_modify.py' in changes.modified
        assert 'unchanged.py' in changes.unchanged
        assert any('module.py' in p for p in changes.unchanged)

        assert changes.has_changes()
        assert changes.total_changed() == 3

    def test_detect_changes_from_snapshot(self):
        """Test detecting changes from saved snapshot."""
        # Create and save initial snapshot
        dag1 = MerkleDAG(str(self.test_path))
        dag1.build()
        self.snapshot_manager.save_snapshot(dag1)

        # Modify files
        (self.test_path / 'to_modify.py').write_text('# modified content')
        (self.test_path / 'new_file.py').write_text('# new')

        # Detect changes from snapshot
        changes, current_dag = self.detector.detect_changes_from_snapshot(str(self.test_path))

        assert 'new_file.py' in changes.added
        assert 'to_modify.py' in changes.modified
        assert len(changes.removed) == 0
        assert changes.has_changes()

    def test_no_changes_detection(self):
        """Test when no changes occur."""
        dag1 = MerkleDAG(str(self.test_path))
        dag1.build()

        dag2 = MerkleDAG(str(self.test_path))
        dag2.build()

        changes = self.detector.detect_changes(dag1, dag2)

        assert not changes.has_changes()
        assert changes.total_changed() == 0
        assert len(changes.unchanged) == 4

    def test_quick_check(self):
        """Test quick change detection."""
        # No snapshot exists - should return True
        assert self.detector.quick_check(str(self.test_path))

        # Save snapshot - excluding snapshots directory
        dag = MerkleDAG(str(self.test_path))
        dag.ignore_patterns.add('snapshots')  # Ignore the snapshots directory
        dag.build()
        self.snapshot_manager.save_snapshot(dag)

        # No changes - should return False
        assert not self.detector.quick_check(str(self.test_path))

        # Make a change
        (self.test_path / 'to_modify.py').write_text('# changed')

        # Should detect change
        assert self.detector.quick_check(str(self.test_path))

    def test_files_to_reindex(self):
        """Test getting files that need reindexing."""
        changes = FileChanges(
            added=['new1.py', 'new2.py'],
            removed=['old.py'],
            modified=['changed.py'],
            unchanged=['same.py']
        )

        files_to_reindex = self.detector.get_files_to_reindex(changes)

        assert len(files_to_reindex) == 3
        assert 'new1.py' in files_to_reindex
        assert 'new2.py' in files_to_reindex
        assert 'changed.py' in files_to_reindex
        assert 'old.py' not in files_to_reindex

    def test_files_to_remove(self):
        """Test getting files to remove from index."""
        changes = FileChanges(
            added=['new.py'],
            removed=['deleted.py'],
            modified=['changed.py'],
            unchanged=['same.py']
        )

        files_to_remove = self.detector.get_files_to_remove(changes)

        assert len(files_to_remove) == 2
        assert 'deleted.py' in files_to_remove
        assert 'changed.py' in files_to_remove  # Modified files need old chunks removed
        assert 'new.py' not in files_to_remove
