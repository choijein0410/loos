"""Frame selection checks using valid DCDs with identifiable coordinates.

Run with a Python environment containing built LOOS bindings:
    python -m unittest discover -s tests -p 'test_pyloos_trajectories.py' -v
"""
from pathlib import Path
import tempfile
import unittest

import loos
from loos import pyloos


class FrameSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.pdb = root / 'model.pdb'
        cls.pdb.write_text(
            'ATOM      1  C1  DUM A   1       0.000   0.000   0.000  1.00  0.00           C\n'
            'ATOM      2  C2  DUM A   1       0.000   1.000   0.000  1.00  0.00           C\n'
            'ATOM      3  C3  DUM A   1       0.000   0.000   1.000  1.00  0.00           C\n'
            'ATOM      4  C4  DUM A   1       1.000   0.000   0.000  1.00  0.00           C\n'
            'END\n'
        )
        cls.files = []
        for name, offset in (('a', 0), ('b', 10)):
            filename = root / (name + '.dcd')
            cls.files.append(filename)
            model = loos.createSystem(str(cls.pdb))
            writer = loos.DCDWriter(str(filename))
            writer.setHeader(4, 6, 0.001, False)
            writer.writeHeader()
            for frame in range(6):
                for atom, (x, y, z) in zip(model, ((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0))):
                    atom.coords(loos.GCoord(offset + frame + x, y, z))
                writer.writeFrame(model)
            del writer

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def trajectory(self, file_index=0, **kwargs):
        model = loos.createSystem(str(self.pdb))
        return pyloos.Trajectory(str(self.files[file_index]), model, **kwargs)

    def virtual(self, **kwargs):
        return pyloos.VirtualTrajectory(self.trajectory(), self.trajectory(1), **kwargs)

    @staticmethod
    def xcoords(trajectory):
        # Read scalars immediately: trajectory groups share live model atoms.
        return [float(frame[0].coords()[0]) for frame in trajectory]

    def test_unfiltered_and_constructor_controls(self):
        self.assertEqual(self.xcoords(self.trajectory()), [0, 1, 2, 3, 4, 5])
        self.assertEqual(self.xcoords(self.trajectory(stride=2)), [0, 2, 4])
        self.assertEqual(self.xcoords(self.trajectory(skip=2)), [2, 3, 4, 5])
        self.assertEqual(self.xcoords(self.trajectory(skip=1, stride=2)), [1, 3, 5])

    def test_stride_setter(self):
        traj = self.trajectory()
        self.assertEqual(len(traj), 6)
        traj.stride(2)
        self.assertEqual(self.xcoords(traj), [0, 2, 4])
        traj.stride(1)
        self.assertEqual(len(traj), 6)

    def test_skip_setter(self):
        traj = self.trajectory()
        self.assertEqual(len(traj), 6)
        traj.skip(2)
        self.assertEqual(self.xcoords(traj), [2, 3, 4, 5])
        traj.skip(0)
        self.assertEqual(len(traj), 6)

    def test_setters_refresh_length_metadata_indexing_and_slicing(self):
        traj = self.trajectory()
        traj.skip(1)
        traj.stride(2)
        self.assertEqual(len(traj), 3)
        self.assertEqual(traj.frameNumber(range(len(traj))), [1, 3, 5])
        self.assertEqual(float(traj[-1][0].coords()[0]), 5)
        self.assertEqual(self.xcoords(traj[::-1]), [5, 3, 1])
        self.assertEqual(float(traj.readFrame(0)[0].coords()[0]), 1)

    def test_explicit_one_shot_iterator_overrides_setters(self):
        traj = self.trajectory(iterator=iter([0, 2, 5]))
        traj.skip(3)
        traj.stride(4)
        self.assertEqual(self.xcoords(traj), [0, 2, 5])
        self.assertEqual(traj.frameNumber(range(len(traj))), [0, 2, 5])

    def test_all_stride_updates_child_selections(self):
        traj = self.virtual()
        self.assertEqual(len(traj), 12)
        traj.allStride(2)
        self.assertEqual(self.xcoords(traj), [0, 2, 4, 10, 12, 14])

    def test_all_skip_updates_child_selections(self):
        traj = self.virtual()
        self.assertEqual(len(traj), 12)
        traj.allSkip(2)
        self.assertEqual(self.xcoords(traj), [2, 3, 4, 5, 12, 13, 14, 15])

    def test_virtual_constructor_skip_stride_and_reverse_slice(self):
        traj = self.virtual(skip=1, stride=2)
        self.assertEqual(self.xcoords(traj), [1, 3, 5, 11, 13, 15])
        self.assertEqual(self.xcoords(traj[::-1]), [15, 13, 11, 5, 3, 1])

    def test_aligned_selection_refreshes_transforms_after_length_query(self):
        ref = loos.createSystem(str(self.pdb))
        traj = pyloos.AlignedVirtualTrajectory(
            self.trajectory(), self.trajectory(1), alignwith='all', reference=ref
        )
        self.assertAlmostEqual(float(traj[0][0].coords()[0]), 0)
        traj.skip(2)
        self.assertEqual(len(traj), 10)
        self.assertAlmostEqual(float(traj[0][0].coords()[0]), 0)
        traj.stride(2)
        self.assertEqual(len(traj), 5)
        for frame in traj[:]:
            self.assertAlmostEqual(float(frame[0].coords()[0]), 0)

    def test_aligned_child_selection_refreshes_transforms(self):
        ref = loos.createSystem(str(self.pdb))
        traj = pyloos.AlignedVirtualTrajectory(
            self.trajectory(), self.trajectory(1), alignwith='all', reference=ref
        )
        self.assertAlmostEqual(float(traj[0][0].coords()[0]), 0)
        traj.allStride(2)
        self.assertEqual(len(traj), 6)
        for frame in traj[:]:
            self.assertAlmostEqual(float(frame[0].coords()[0]), 0)

    def test_aligned_child_skip_refreshes_transforms(self):
        ref = loos.createSystem(str(self.pdb))
        traj = pyloos.AlignedVirtualTrajectory(
            self.trajectory(), self.trajectory(1), alignwith='all', reference=ref
        )
        traj[0]
        traj.allSkip(2)
        self.assertEqual(len(traj), 8)
        for frame in traj[:]:
            self.assertAlmostEqual(float(frame[0].coords()[0]), 0)

    def test_iterative_alignment_matches_fresh_selected_trajectory(self):
        traj = pyloos.AlignedVirtualTrajectory(
            self.trajectory(), self.trajectory(1), alignwith='all'
        )
        traj[0]
        traj.skip(2)
        traj.stride(2)
        self.assertEqual(len(traj), 5)
        fresh = pyloos.AlignedVirtualTrajectory(
            self.trajectory(), self.trajectory(1), alignwith='all', skip=2, stride=2
        )
        actual = traj[:]
        expected = fresh[:]
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            self.assertEqual(len(got), len(want))
            for got_atom, want_atom in zip(got, want):
                for axis in range(3):
                    self.assertAlmostEqual(got_atom.coords()[axis], want_atom.coords()[axis])

    def test_aligned_selection_refreshes_before_direct_indexing(self):
        ref = loos.createSystem(str(self.pdb))
        first = self.trajectory()
        traj = pyloos.AlignedVirtualTrajectory(
            first, self.trajectory(1), alignwith='all', reference=ref
        )
        traj[0]
        traj.skip(2)
        traj[0]
        self.assertEqual(first.trajectory().currentFrame(), 2)
        traj.allStride(2)
        self.assertAlmostEqual(float(traj[0][0].coords()[0]), 0)


if __name__ == '__main__':
    unittest.main()
