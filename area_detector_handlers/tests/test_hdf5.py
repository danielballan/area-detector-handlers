from area_detector_handlers.tests.conftest import select_handler
import numpy as np


@select_handler("AD_HDF5")
def test_hdf5(hdf5_files, handler):
    (rpath, kwargs), (N_rows, N_cols, N_points, fpp) = hdf5_files
    expected_shape = (fpp, N_rows, N_cols)
    with handler(rpath, **kwargs) as h:
        for frame in range(N_points):
            d = h(point_number=frame)
            assert d.shape == expected_shape
            assert np.all(d == frame)


@select_handler("AD_HDF5")
def test_close_one(hdf5_files, handler):
    "Aim two handlers at the same file. Close one; ensure the other still works."
    (rpath, kwargs), (N_rows, N_cols, N_points, fpp) = hdf5_files
    handler1 = handler(rpath, **kwargs)
    handler2 = handler(rpath, **kwargs)
    np.asarray(handler1(point_number=0))
    assert handler1._file is not None
    np.asarray(handler2(point_number=0))
    assert handler2._file is not None
    handler1.close()
    assert handler1._file is None
    assert handler2._file is not None
    assert not handler2._file.closed
    for frame in range(N_points):
        np.asarray(handler2(point_number=frame))
