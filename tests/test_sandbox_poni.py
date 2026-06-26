from pathlib import Path

from tabs.sandbox_poni import parse_poni_file


def test_parse_poni_file_reads_geometry_and_wavelength(tmp_path):
    poni_path = tmp_path / "calibration.poni"
    poni_path.write_text(
        """# Calibration\n"
        "PixelSize1: 0.000103358\n"
        "PixelSize2: 0.00010253\n"
        "Distance: 0.116859855632\n"
        "Poni1: 0.0529565255372\n"
        "Poni2: 0.05473342483\n"
        "Rot1: 0.015821123969\n"
        "Wavelength: 7.084811024e-11\n"
        """,
        encoding="utf-8",
    )

    metadata = parse_poni_file(poni_path)

    assert metadata["center_x_px"] == 0.0529565255372
    assert metadata["center_y_px"] == 0.05473342483
    assert metadata["distance_m"] == 0.116859855632
    assert metadata["pixel_size_x_m"] == 0.000103358
    assert metadata["pixel_size_y_m"] == 0.00010253
    assert metadata["wavelength_m"] == 7.084811024e-11
