"""Cocotb lane-equivalence test for ssm_unit.

Builds the tb_ssm_lanes_top harness (one LANES=1 instance and one LANES=4
instance), loads the same random A_log/D/dt_bias into both, streams the same
random frames into both and asserts that the two output sequences are
bit-identical (data and tlast). The lane optimization must preserve the exact
per-element arithmetic order, so any mismatch is a bug.

The two instances are not cycle-aligned (LANES=4 processes a dim block in the
same cycles LANES=1 needs for one dim), so input beats are only issued when
both instances are ready and each output sequence is collected independently;
the sequences are then compared element by element.

Geometry comes from the build via SSM_HEADS / SSM_HEAD_DIM / SSM_D_STATE
(default 3/5/6: a full 4-lane block plus a masked partial block).
"""

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

NH = int(os.environ.get("SSM_HEADS", "3"))
HD = int(os.environ.get("SSM_HEAD_DIM", "5"))
DS = int(os.environ.get("SSM_D_STATE", "6"))
XN = NH * HD
FRAME = XN + 2 * DS + NH
NFRAMES = 6
SEED = 0x55AA


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.load_en.value = 0
    dut.load_sel.value = 0
    dut.load_idx.value = 0
    dut.load_wdata.value = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    dut.s_axis_tlast.value = 0
    dut.m0_axis_tready.value = 0
    dut.m1_axis_tready.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def send_beat(dut, data, last):
    """One input beat, accepted by both instances in the same cycle.

    The instances become ready at different times (LANES=4 finishes a token
    sooner), so the beat is held until both tready signals are high; they then
    consume it on the same clock edge and stay on identical input streams.
    """
    dut.s_axis_tvalid.value = 1
    dut.s_axis_tdata.value = data
    dut.s_axis_tlast.value = 1 if last else 0
    for _ in range(100000):
        if int(dut.s0_axis_tready.value) and int(dut.s1_axis_tready.value):
            break
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    else:
        try:
            dbg = (f"state0={int(dut.u_lanes1.state.value)} "
                   f"h0={int(dut.u_lanes1.h_cnt.value)} "
                   f"d0={int(dut.u_lanes1.d_base.value)} "
                   f"t0={int(dut.u_lanes1.t_cnt.value)} "
                   f"o0={int(dut.u_lanes1.out_cnt.value)} "
                   f"state1={int(dut.u_lanes4.state.value)} "
                   f"h1={int(dut.u_lanes4.h_cnt.value)} "
                   f"d1={int(dut.u_lanes4.d_base.value)} "
                   f"t1={int(dut.u_lanes4.t_cnt.value)} "
                   f"o1={int(dut.u_lanes4.out_cnt.value)}")
        except Exception as exc:  # pragma: no cover - diagnostics only
            dbg = f"internal signals unavailable ({exc})"
        raise AssertionError(
            "input beats never accepted: "
            f"s0_tready={int(dut.s0_axis_tready.value)} "
            f"s1_tready={int(dut.s1_axis_tready.value)} "
            f"busy0={int(dut.busy0.value)} busy1={int(dut.busy1.value)}; "
            + dbg
        )
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_ssm_lanes(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    rng = random.Random(SEED)

    dut._log.info("Loading identical random weights into both instances ...")
    for sel in (0, 1, 2):
        for i in range(NH):
            dut.load_en.value = 1
            dut.load_sel.value = sel
            dut.load_idx.value = i
            dut.load_wdata.value = rng.getrandbits(16)
            await RisingEdge(dut.clk)
    dut.load_en.value = 0

    total = 0
    for t in range(NFRAMES):
        frame = [rng.getrandbits(16) for _ in range(FRAME)]
        for i, data in enumerate(frame):
            await send_beat(dut, data, i == FRAME - 1)
        dut.s_axis_tvalid.value = 0
        dut.s_axis_tlast.value = 0

        got0, got1 = [], []
        last0, last1 = [], []
        dut.m0_axis_tready.value = 1
        dut.m1_axis_tready.value = 1
        for _ in range(XN * 200 + 2000):
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            if int(dut.m0_axis_tvalid.value):
                got0.append(int(dut.m0_axis_tdata.value) & 0xFFFFFFFF)
                last0.append(int(dut.m0_axis_tlast.value))
            if int(dut.m1_axis_tvalid.value):
                got1.append(int(dut.m1_axis_tdata.value) & 0xFFFFFFFF)
                last1.append(int(dut.m1_axis_tlast.value))
            if not int(dut.busy0.value) and not int(dut.busy1.value):
                break
        dut._log.info(
            f"frame {t}: LANES=1 {len(got0)} beats, LANES=4 {len(got1)} beats"
        )
        assert len(got0) == XN, f"frame {t}: LANES=1 produced {len(got0)}/{XN}"
        assert len(got1) == XN, f"frame {t}: LANES=4 produced {len(got1)}/{XN}"
        for i in range(XN):
            assert got0[i] == got1[i], (
                f"frame {t} beat {i}: LANES=1 {got0[i]:08x} != "
                f"LANES=4 {got1[i]:08x}"
            )
            assert last0[i] == last1[i], f"frame {t} beat {i}: tlast diverged"
        total += XN
        dut.m0_axis_tready.value = 0
        dut.m1_axis_tready.value = 0
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)

    dut._log.info(
        f"Lane equivalence: {total}/{total} output beats bit-identical "
        f"(LANES=1 vs LANES=4, {NFRAMES} frames, {NH}/{HD}/{DS})"
    )
