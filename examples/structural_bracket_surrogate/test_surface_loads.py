"""Tests for physically consistent boundary-face traction."""
import numpy as np
import pytest
from surface_loads import exterior_faces, plane_faces, traction_nodal_forces, bracket_boundary_case


def test_single_tetra_faces_and_uniform_traction():
    nodes = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    tets = np.array([[0,1,2,3]])
    faces = exterior_faces(tets)
    assert len(faces) == 4
    bottom = plane_faces(nodes, faces, axis=2, value=0.)
    assert len(bottom) == 1
    forces, area = traction_nodal_forces(nodes, bottom, np.array([0.,0.,10.]))
    assert area == pytest.approx(0.5)
    assert forces.sum(axis=0) == pytest.approx([0.,0.,5.])
    assert np.allclose(forces[3], 0)


def test_two_tetra_shared_face_is_internal():
    tets = np.array([[0,1,2,3],[0,1,2,4]])
    faces = exterior_faces(tets)
    assert len(faces) == 6
    assert not any(np.array_equal(np.sort(face), [0,1,2]) for face in faces)


def test_zero_face_selection_rejected():
    with pytest.raises(ValueError):
        traction_nodal_forces(np.zeros((4,3)), np.empty((0,3), int),
                              np.array([1.,0.,0.]))


def test_bracket_bc_has_nonzero_resultant():
    # Unit cube with conforming six-tet decomposition along main diagonal.
    nodes = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
                      [0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=float)
    tets = np.array([[0,1,3,7],[0,3,2,7],[0,2,6,7],
                     [0,6,4,7],[0,4,5,7],[0,5,1,7]])
    fixed, forces, metadata = bracket_boundary_case(nodes, tets, (2.,0.,0.))
    assert len(fixed) == 4
    assert np.allclose(forces.sum(axis=0), [2.,0.,0.])
    assert metadata["loaded_area_m2"] == pytest.approx(1.0)
