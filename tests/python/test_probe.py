import copyengine as ce


def test_probe_shape():
    r = ce.probe()
    assert r["copyengine"] == ce.__version__
    cuda = r["cuda"]
    assert isinstance(cuda["available"], bool)
    if not cuda["available"]:
        assert cuda["reason"]
        return
    for dev in cuda["devices"]:
        for name, feat in dev["features"].items():
            assert isinstance(feat["ok"], bool), name
            assert feat["ok"] or feat["reason"], name
        # Plain async H2D/D2H must work on any usable CUDA device.
        assert dev["features"]["async_copy"]["ok"], dev["features"]["async_copy"]
