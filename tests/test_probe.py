from board_opt.models import HardwareProfile
from board_opt.probe import parse_vulkaninfo, select_native_formats, compute_usable_vram_mb


def test_parse_vulkaninfo_bc250():
    sample = (
        "deviceName = AMD BC-250 (RADV GFX1013)\n"
        "deviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU\n"
    )
    info = parse_vulkaninfo(sample)
    assert "GFX1013" in info["gpu_name"]
    assert info["uma"] is True


def test_select_native_formats_no_intdot_no_fp4():
    hw = HardwareProfile(
        cpu_arch="x86_64",
        cpu_march="znver2",
        cpu_cores=6,
        ram_total_mb=16384,
        usable_vram_mb=14336,
        backend="vulkan",
        gpu_name="AMD BC-250 (RADV GFX1013)",
        uma=True,
        fp16=True,
        bf16=False,
        fp4=False,
        int_dot=False,
        preferred_formats=[],
    )
    assert select_native_formats(hw) == ["Q4_0", "Q8_0", "F16"]


def test_compute_usable_vram_mb_default_reserve():
    # max(1536, 10% of 16384=1638) = 1638 -> 16384 - 1638 = 14746
    assert compute_usable_vram_mb(total_mb=16384, reserve_mb=None) == 14746
    assert compute_usable_vram_mb(total_mb=16384, reserve_mb=2048) == 14336
