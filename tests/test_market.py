"""Tests for hostai.market: query, filtering, scoring, and hardware rank."""


import pytest

from hostai import market, utils
from hostai.profiles import (
    HardwareRank,
    MarketPolicy,
    MonitorHardware,
    Profile,
    Profiles,
)


def make_profile(disk_gb=None, query="gpu_name == RTX_4090"):
    return Profile(
        name="test",
        image="a",
        ctx_size=32768,
        gpu_query=query,
        disk_gb=disk_gb,
    )


def test_build_search_query_includes_zero_traffic_costs(config):
    """A zero max transfer cost must still be written into the query."""
    config.market.max_inet_down_cost = 0.0
    config.market.max_inet_up_cost = 0.0
    config.market.disk_gb = 80
    profiles = Profiles(
        schema_version=1,
        images=[],
        profiles=[make_profile()],
        monitor_hardware=MonitorHardware(),
        market_policy=MarketPolicy(require_free_traffic=True),
    )
    query, _ = market.build_search_query(config, profiles, make_profile())
    assert "inet_down_cost<=0.0" in query
    assert "inet_up_cost<=0.0" in query
    assert "disk_space>=80" in query


def test_build_search_query_appends_max_dph_when_offer_not_given(config):
    profiles = Profiles(
        schema_version=1,
        images=[],
        profiles=[make_profile()],
        monitor_hardware=MonitorHardware(),
        market_policy=MarketPolicy(require_free_traffic=False),
    )
    query, max_dph = market.build_search_query(config, profiles, make_profile(), max_price=0.4)
    assert "dph_total <= 0.4" in query
    assert max_dph == 0.4


def test_build_search_query_replaces_stale_disk_space(config):
    """Any disk_space in the profile query is replaced by the resolved disk_gb."""
    config.market.disk_gb = 80
    profiles = Profiles(
        schema_version=1,
        images=[],
        profiles=[make_profile(query="gpu_name == RTX_4090 disk_space>=200")],
        monitor_hardware=MonitorHardware(),
        market_policy=MarketPolicy(require_free_traffic=False),
    )
    query, _ = market.build_search_query(config, profiles, profiles.profiles[0])
    assert "disk_space>=200" not in query
    assert "disk_space>=80" in query
    # Only one disk_space constraint should remain.
    assert query.count("disk_space") == 1


def test_resolved_disk_gb_uses_profile_override(config):
    profile = make_profile(disk_gb=175)
    assert market.resolved_disk_gb(profile, config) == 175


def test_resolved_disk_gb_falls_back_to_market(config):
    config.market.disk_gb = 80
    assert market.resolved_disk_gb(make_profile(), config) == 80


def test_resolved_disk_gb_independent_of_ctx_size(config):
    """Large context alone must not inflate the persistent disk requirement."""
    config.market.disk_gb = 35
    big_ctx = make_profile(disk_gb=None, query="gpu_name == RTX_4090")
    big_ctx.ctx_size = 262144
    assert market.resolved_disk_gb(big_ctx, config) == 35


def test_is_same_or_better_gpu_rejects_worse_rank():
    profiles = Profiles(
        schema_version=1,
        images=[],
        profiles=[],
        monitor_hardware=MonitorHardware(
            policy="same_or_better",
            gpu_ranks=[
                HardwareRank(gpu="A40", aliases=["A40"], rank=100),
                HardwareRank(gpu="RTX 4090", aliases=["RTX_4090"], rank=200),
            ],
        ),
        market_policy=MarketPolicy(),
    )
    assert market.is_same_or_better_gpu(profiles, "RTX 4090", "A40") is False
    assert market.is_same_or_better_gpu(profiles, "A40", "RTX 4090") is True
    assert market.is_same_or_better_gpu(profiles, "RTX 4090", "RTX 4090") is True


def test_filter_eligible_offers_respects_max_dph_and_gpu_rank(config):
    profiles = Profiles(
        schema_version=1,
        images=[],
        profiles=[],
        monitor_hardware=MonitorHardware(
            policy="same_or_better",
            gpu_ranks=[
                HardwareRank(gpu="A40", aliases=["A40"], rank=100),
                HardwareRank(gpu="RTX 4090", aliases=["RTX_4090"], rank=200),
            ],
        ),
        market_policy=MarketPolicy(),
    )
    offers = [
        {"id": 1, "gpu_name": "A40", "dph_total": 0.3},
        {"id": 2, "gpu_name": "RTX 4090", "dph_total": 0.5},
        {"id": 3, "gpu_name": "RTX 4090", "dph_total": 0.9},
    ]
    matches = market.filter_eligible_offers(offers, max_dph=0.8, current_gpu="RTX 4090", profiles=profiles)
    assert [o["id"] for o in matches] == [2]


def test_filter_eligible_offers_zero_max_dph():
    """max_dph=0 must still filter out positive prices."""
    offers = [
        {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.0},
        {"id": 2, "gpu_name": "RTX 4090", "dph_total": 0.01},
    ]
    matches = market.filter_eligible_offers(offers, max_dph=0.0)
    assert [o["id"] for o in matches] == [1]


def test_score_offers_dph_mode_sorts_by_price(config):
    config.market.scoring_mode = "dph"
    offers = [
        {"id": 1, "dph_total": 0.5},
        {"id": 2, "dph_total": 0.3},
        {"id": 3, "dph_total": 0.7},
    ]
    scored = market.score_offers(offers, config, {})
    assert [o["id"] for o in scored] == [2, 1, 3]


def test_score_offers_perf_mode_with_history(config):
    """With historical TPS, cheaper-per-token offer should win."""
    config.market.scoring_mode = "perf"
    config.market.scoring_prompt_weight = 0.0
    config.market.scoring_decode_weight = 1.0
    config.market.min_historical_samples = 1
    offers = [
        {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.5, "inet_down": 1000},
        {"id": 2, "gpu_name": "RTX 3090", "dph_total": 0.3, "inet_down": 1000},
    ]
    stats = {
        "rtx4090": {"prompt_tps": 1000, "prompt_samples": 1, "decode_tps": 2000, "decode_samples": 1},
        "rtx3090": {"prompt_tps": 1000, "prompt_samples": 1, "decode_tps": 1000, "decode_samples": 1},
    }
    scored = market.score_offers(offers, config, stats)
    # At 0.5 $/h and 2000 tps vs 0.3 $/h and 1000 tps: cost/token = 0.5/2000=0.00025 vs 0.3/1000=0.0003
    assert scored[0]["id"] == 1


def test_score_offers_session_mode_prefers_short_startup(config):
    """A faster expected startup should give a lower total session cost."""
    config.market.scoring_mode = "session"
    config.market.scoring_prompt_weight = 0.0
    config.market.scoring_decode_weight = 1.0
    config.market.min_historical_samples = 1
    config.market.model_download_gb = 5.0
    config.market.image_size_gb = 0.0
    offers = [
        {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.5, "inet_down": 100, "inet_down_cost": 0.0},
        {"id": 2, "gpu_name": "RTX 4090", "dph_total": 0.5, "inet_down": 1000, "inet_down_cost": 0.0},
    ]
    stats = {
        "rtx4090": {"prompt_tps": 1000, "prompt_samples": 1, "decode_tps": 1000, "decode_samples": 1},
    }
    scored = market.score_offers(offers, config, stats, session_seconds=300)
    # Offer 2 has faster download (same dph), so lower total cost / tokens.
    assert scored[0]["id"] == 2


def test_offer_download_estimate_uses_bottleneck(config):
    """Startup must wait for the slower of network transfer and disk processing."""
    config.market.model_download_gb = 18.83
    config.market.image_size_gb = 5.0
    # Fast network, slow disk: disk is the bottleneck.
    offer = {"inet_down": 1000, "disk_bw": 10}
    size, seconds, _ = market.offer_download_estimate(offer, config)
    net_seconds = (size * 8 * 1000) / 1000
    disk_seconds = (size * 1024 / 10) * 2
    assert seconds == pytest.approx(max(30.0, net_seconds, disk_seconds), rel=1e-3)
    assert seconds > net_seconds

    # Fast disk, slow network: network is the bottleneck.
    offer = {"inet_down": 10, "disk_bw": 1000}
    size, seconds, _ = market.offer_download_estimate(offer, config)
    net_seconds = (size * 8 * 1000) / 10
    disk_seconds = (size * 1024 / 1000) * 2
    assert seconds == pytest.approx(max(30.0, net_seconds, disk_seconds), rel=1e-3)
    assert seconds > disk_seconds

    # Missing bandwidth data falls back to the minimum startup floor.
    offer = {"inet_down": 0, "disk_bw": 0}
    size, seconds, _ = market.offer_download_estimate(offer, config)
    assert seconds == 30.0


def test_build_search_query_bid_price_caps_max_dph(config):
    """A bid price below max_dph must lower the search ceiling."""
    config.market.max_dph = 0.8
    config.market.disk_gb = 35
    profiles = Profiles(
        schema_version=1,
        images=[],
        profiles=[make_profile()],
        monitor_hardware=MonitorHardware(),
        market_policy=MarketPolicy(require_free_traffic=False),
    )
    query, max_dph = market.build_search_query(config, profiles, make_profile(), bid_price=0.35)
    assert "dph_total <= 0.35" in query
    assert max_dph == 0.35


def test_build_search_query_bid_price_ignored_when_none(config):
    """Without a bid, max_dph is the configured ceiling."""
    config.market.max_dph = 0.8
    config.market.disk_gb = 35
    profiles = Profiles(
        schema_version=1,
        images=[],
        profiles=[make_profile()],
        monitor_hardware=MonitorHardware(),
        market_policy=MarketPolicy(require_free_traffic=False),
    )
    query, max_dph = market.build_search_query(config, profiles, make_profile())
    assert "dph_total <= 0.8" in query
    assert max_dph == 0.8


def test_score_offers_perf_mode_without_history_uses_conservative_fallback(config):
    """Unknown GPUs in perf mode must still be scored in $/token units."""
    config.market.scoring_mode = "perf"
    config.market.scoring_prompt_weight = 0.0
    config.market.scoring_decode_weight = 1.0
    offers = [
        {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.5},
        {"id": 2, "gpu_name": "RTX 3090", "dph_total": 0.3},
    ]
    scored = market.score_offers(offers, config, {})
    # Both use the same conservative fallback, so the cheaper GPU should still
    # win because the score is (dph/3600) * (1/tps_fallback).
    assert scored[0]["id"] == 2
    assert all("_hostai_score" in o for o in scored)


def test_score_offers_perf_mode_known_better_than_unknown(config):
    """A known fast GPU should beat an unknown GPU with only a small price premium."""
    config.market.scoring_mode = "perf"
    config.market.scoring_prompt_weight = 0.0
    config.market.scoring_decode_weight = 1.0
    config.market.min_historical_samples = 1
    offers = [
        {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.5},
        {"id": 2, "gpu_name": "RTX 3090", "dph_total": 0.45},
    ]
    stats = {"rtx4090": {"decode_tps": 2000, "decode_samples": 1}}
    scored = market.score_offers(offers, config, stats)
    # RTX 4090 cost/token = 0.5/3600/2000 ~ 6.9e-8.
    # RTX 3090 unknown fallback 50 tps => 0.45/3600/50 = 2.5e-6, much worse.
    assert scored[0]["id"] == 1


def test_offer_download_estimate_cache_states_differ(config):
    """Cold, cached, and unknown cache states must give distinct download sizes."""
    offer = {"inet_down": 1000, "disk_bw": 1000, "inet_down_cost": 0.01}
    cold = market.offer_download_estimate(offer, config, cache_state=market.CACHE_STATE_COLD)
    cached = market.offer_download_estimate(offer, config, cache_state=market.CACHE_STATE_CACHED)
    unknown = market.offer_download_estimate(offer, config, cache_state=market.CACHE_STATE_UNKNOWN)
    assert cold[0] > unknown[0] > cached[0]
    assert cold[2] > unknown[2] > cached[2]


def test_session_scoring_uses_cache_state(config):
    """A persistent volume should reduce the session score by skipping the model download."""
    config.market.scoring_mode = "session"
    config.vast.volume_id = "vol-123"
    config.vast.volume_mount_path = "/models"
    offer = {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.5, "inet_down": 1000, "disk_bw": 1000, "inet_down_cost": 0.0}
    scored_cached = market.score_offers([dict(offer)], config, {}, session_seconds=300)

    config.vast.volume_id = ""
    config.vast.volume_mount_path = ""
    scored_cold = market.score_offers([dict(offer)], config, {}, session_seconds=300)

    assert scored_cached[0]["_hostai_score"].score < scored_cold[0]["_hostai_score"].score
    assert scored_cached[0]["_hostai_score"].startup_seconds < scored_cold[0]["_hostai_score"].startup_seconds


def test_session_scoring_total_cost_units(config):
    """Session scoring must return total cost, not cost per token."""
    config.market.scoring_mode = "session"
    offer = {"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.5, "inet_down": 1000, "disk_bw": 1000, "inet_down_cost": 0.0}
    stats = {"rtx4090": {"decode_tps": 1000, "decode_samples": 1}}
    scored = market.score_offers([offer], config, stats, session_seconds=3600)
    # Total cost for one hour at $0.5/h plus a small startup cost.
    assert scored[0]["_hostai_score"].score >= 0.5
    assert scored[0]["_hostai_score"].score < 1.0


def test_parse_duration_to_seconds():
    assert utils.parse_duration_to_seconds("30m") == 1800
    assert utils.parse_duration_to_seconds("2h") == 7200
    assert utils.parse_duration_to_seconds("") is None
    with pytest.raises(ValueError):
        utils.parse_duration_to_seconds("abc")
