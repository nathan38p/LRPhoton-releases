import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabs import radial_tab


def test_parse_edf_header_supports_colon_delimited_keys():
    header_text = """{
DataType = FloatValue ;
ByteOrder = LowByteFirst ;
Dim_1 = 1028 ;
Dim_2 = 1062 ;
Center_1 = 612.199 ;
Center_2 = 647.162 ;
PSize_1 = 7.5e-05 ;
PSize_2 = 7.5e-05 ;
WaveLength = 1.54189e-10 ;
SampleDistance = 0.9 ;
}"""

    header = radial_tab.parse_edf_header(header_text)

    assert header["Center_1"] == "612.199"
    assert header["Center_2"] == "647.162"
    assert header["PSize_1"] == "7.5e-05"
    assert header["PSize_2"] == "7.5e-05"
    assert header["WaveLength"] == "1.54189e-10"
    assert header["SampleDistance"] == "0.9"


def test_parse_edf_header_supports_legacy_key_value_pairs():
    header_text = """{
DataType: FloatValue ;
ByteOrder: LowByteFirst ;
Dim_1: 1028 ;
Dim_2: 1062 ;
Center_1: 612.199 ;
Center_2: 647.162 ;
PSize_1: 7.5e-05 ;
PSize_2: 7.5e-05 ;
WaveLength: 1.54189e-10 ;
SampleDistance: 0.9 ;
}"""

    header = radial_tab.parse_edf_header(header_text)

    assert header["Center_1"] == "612.199"
    assert header["Center_2"] == "647.162"
    assert header["PSize_1"] == "7.5e-05"
    assert header["PSize_2"] == "7.5e-05"
    assert header["WaveLength"] == "1.54189e-10"
    assert header["SampleDistance"] == "0.9"


def test_header_q_geometry_values_fallback_to_theoretical_center():
    header = {
        "Theoretical_Center_1": "607.031",
        "Theoretical_Center_2": "651.003",
        "PSize_1": "7.5e-05",
        "PSize_2": "7.5e-05",
        "WaveLength": "1.54189e-10",
        "SampleDistance": "0.9",
    }

    values, missing = radial_tab.header_q_geometry_values(header)

    assert values["cx"] == 607.031
    assert values["cy"] == 651.003
    assert missing == []


def test_q_from_detector_geometry_uses_half_angle():
    q_value = radial_tab.q_from_detector_geometry(0.1, 1.0, 1.54189)
    expected = (4.0 * 3.141592653589793 / (1.54189 * 0.1)) * 0.04995837495788057

    assert abs(q_value - expected) < 1e-12
