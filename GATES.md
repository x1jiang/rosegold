# Acceptance Gates: Dynamic Hardware Auto-Selection (GPU vs CPU)

## Gate 1: Dynamic Hardware Detection & Model Pairing
- [x] Implement `app/model_selector.py` which automatically inspects `torch.cuda.is_available()`, VRAM, and pairs CPU to lightweight models (`Llama-3.2-3B-Instruct` / `gemma-2-2b-it`) and GPU to full models (`Llama-3.1-8B-Instruct` / `gemma-2-9b-it` / `muse-glimmer-30b`).
  CHECK: `/Users/xiaoqianjiang/anaconda3/bin/python -m pytest tests/test_model_selector.py -v`
  EXPECT: `3 passed`

## Gate 2: Full Engine & API Auto-Routing
- [x] Engine, CLI, FastAPI, and Web UI automatically detect hardware and route execution to vLLM (on GPU) or CPUEngine (on CPU) with zero manual configuration required.
  CHECK: `/Users/xiaoqianjiang/anaconda3/bin/python -m pytest tests/test_api.py -v`
  EXPECT: `6 passed`

## Gate 3: Master Test Suite Integrity
- [x] All tests pass across GPU routing, CPU lightweight routing, API, UI, and loader/config helpers.
  CHECK: `/Users/xiaoqianjiang/anaconda3/bin/python -m pytest tests/ -q`
  EXPECT: all passed
