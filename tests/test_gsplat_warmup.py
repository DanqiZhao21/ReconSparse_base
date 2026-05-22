from framework.utils.gsplat_warmup import build_gsplat_warmup_cmd


def test_gsplat_warmup_uses_legacy_backend_for_reconsimulator_render_path():
    cmd = build_gsplat_warmup_cmd("python")
    joined = " ".join(cmd)
    assert "ensure_gsplat_legacy_backend" in joined
    assert "ensure_gsplat_backend" not in joined
