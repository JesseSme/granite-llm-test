"""Debug test: check fp_exp and basic operation."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import struct

def f2h(v):
    return struct.unpack('I', struct.pack('f', v))[0]

def h2f(h):
    return struct.unpack('f', struct.pack('I', h & 0xFFFFFFFF))[0]

@cocotb.test()
async def test_debug(dut):
    clock = Clock(dut.clk, 10, unit='ns')
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    dut.valid_in.value = 0
    dut.data_in.value = 0
    dut.last_in.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # Feed a single row of 8 zeros
    for i in range(8):
        dut.data_in.value = f2h(0.0)
        dut.valid_in.value = 1
        dut.last_in.value = 1 if i == 7 else 0
        await RisingEdge(dut.clk)
    dut.valid_in.value = 0
    dut.last_in.value = 0
    
    # Monitor internal signals while waiting for done
    outputs = []
    for cycle in range(500):
        await RisingEdge(dut.clk)
        
        # Print state transitions
        state = dut.state.value.to_unsigned()
        cnt = dut.cnt.value.to_unsigned()
        
        if cycle < 100 or cycle % 20 == 0:
            dut._log.info(f"cycle={cycle} state={state} cnt={cnt} "
                         f"running_max=0x{dut.running_max.value.to_unsigned():08X} "
                         f"sum=0x{dut.sum.value.to_unsigned():08X} "
                         f"valid_out={dut.valid_out.value} done={dut.done.value}")
        
        if dut.valid_out.value == 1:
            val = h2f(dut.data_out.value.to_unsigned())
            outputs.append(val)
            dut._log.info(f"  OUTPUT: 0x{dut.data_out.value.to_unsigned():08X} = {val:.6f}")
        
        if dut.done.value == 1:
            dut._log.info(f"DONE at cycle {cycle}")
            break
    
    dut._log.info(f"Collected {len(outputs)} outputs")
    for i, v in enumerate(outputs):
        dut._log.info(f"  [{i}] = {v:.6f} (expected 0.125)")
    
    if len(outputs) == 8:
        total = sum(outputs)
        dut._log.info(f"Sum = {total:.6f} (expected 1.0)")
