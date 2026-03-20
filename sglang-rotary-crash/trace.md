# amdpilot — sglang-rotary-crash

| Property | Value |
|----------|-------|
| **Task** | RotaryEmbedding crashes on ROCm with CUDA_HOME not found error |
| **Type** | bugfix |
| **Duration** | 0.1h (325s) |
| **Best metric** | 100.00 score |
| **Trials** | 1 (0 failed) |
| **Stages** | 1 |

---

## Trial 1 — stage1_fix_rotary_embedding_rocm_dispatch [PASS]

**Metric**: 100.00 score  (verified PASS)
**Elapsed**: 4m 18s
**Supervisor** (LLM): `stop` — The task is complete. The agent successfully fixed the RotaryEmbedding ROCm/HIP dispatch issue by adding a forward_hip() method that routes to forward_native() instead of forward_cuda(), fixing the use_fallback_kernel condition to be CUDA-specific, and adding proper device handling in forward_native(). All 3 test checks pass with SCORE: 100.0.
**Nudge agent**: 4 checks, all NOOP (agent on track)

### Agent trajectory

1. Let me start by understanding the task. I need to fix a RotaryEmbedding crash
2. `Used ReadFile (task_description.md)`
3. `Used ReadFile (optimization_state.json)`
4. `Used ReadFile (test_harness.py)`
5. Now let me first reproduce the issue by running the test harness, and then exa
6. `Used ReadFile (sglang/python/sglang/srt/...ayers/rotary_embedding.py)`
7. `Used Shell (/opt/venv/bin/python3 /wo...pace/test_harness.py 2>&1)`
8. The test is failing with a JIT compilation error. The error mentions --expt-re
9. (not (_is_cuda) or self.head_size not in [64, 128, 256, 512]) - this is True
10. AND not cpu
11. AND not xpu
12. AND not npu
13. AND not musa
14. _is_cuda is False
15. So not (_is_cuda) is True
16. So the first part (not (_is_cuda) or ...) is True regardless of head_size
17. And if not cpu/xpu/npu/musa, then the whole condition is True
18. So on HIP, it WILL use the fallback kernel!
19. The test is failing with a JIT compilation error. The error shows --expt-relax
20. `Used ReadFile (sglang/python/sglang/jit_kernel/pos_enc.py)`
21. `Used ReadFile (sglang/python/sglang/srt/layers/utils.py)`
22. I found the issue! In /workspace/sglang/python/sglang/jit_kernel/pos_enc.py, l
23. Found the issue! The pos_enc.py file uses cuda_files in the JIT loading. Let m
24. `Used ReadFile (sglang/python/sglang/jit_kernel/utils.py)`
25. `Used Glob (**/jit_kernel/**/*.py)`
26. Now I understand the issue better. The load_jit function in utils.py always pa
27. `Used Glob (sglang/python/sglang/jit_kernel/*.py)`
28. `Used Grep (hip_files|is_hip)`
29. Let me look at the multi_platform.py file which likely contains the MultiPlatf
30. `Used ReadFile (sglang/python/sglang/srt/...s/utils/multi_platform.py)`
31. `Used ReadFile (sglang/python/sglang/jit_kernel/rope.py)`
32. Now I understand the issue better. The problem is:
33. Let me check how apply_rope_with_cos_sin_cache_inplace handles HIP and look fo
34. `Used Grep (is_hip.*apply_rope|apply_rope.*hip)`
35. `Used ReadFile (sglang/python/sglang/srt/utils/__init__.py)`
36. Let me check the common utils file to see the is_hip function and other utilit
37. `Used ReadFile (sglang/python/sglang/srt/utils/common.py)`
38. Now I understand the issue better. Let me look at the actual fix needed. The p
39. Let me check how apply_rope_with_cos_sin_cache_inplace handles HIP and look fo
40. `Used Grep (hip_files)`
41. `Used Grep (DEFAULT_CUDA_CFLAGS|extra_cuda_cflags)`
42. The issue is clear now. The load_jit function in utils.py always passes DEFAUL
43. NOT cuda OR head_size not in list
44. AND not cpu
45. AND not xpu
46. AND not npu
47. AND not musa
48. _is_cuda is False, so not (_is_cuda) is True
49. So the first part is always True regardless of head_size
50. HIP isn't cpu, xpu, npu, or musa, so the whole condition evaluates to True
51. `Used Grep (is_hip|_is_hip)`
52. Let me look at the rope.py file more closely to see how it handles the JIT com
53. `Used ReadFile (sglang/python/sglang/jit_kernel/rope.py)`
54. I see - rope.py also uses cuda_files in the JIT loading. Both rope.py and pos_
55. Let me understand the full flow better. Let me check if there's any HIP-specif
56. `Used Grep (use_fallback_kernel|fallback_rotary_embedding)`
57. Now I understand the full picture:
58. If not self.use_fallback_kernel: calls apply_rope_with_cos_sin_cache_inpla
59. Else: calls self.fallback_rotary_embedding from pos_enc.py
60. Current buggy condition causes HIP to ALWAYS use fallback (because not (_i
61. But the fallback uses CUDA JIT which fails on HIP
62. Now I understand the issue clearly. Both the fallback path (pos_enc.py) and th
63. `Used StrReplaceFile (sglang/python/sglang/srt/...ayers/rotary_embedding.py)`
64. Now I also need to fix the condition that determines use_fallback_kernel. The
65. Now let me also fix the condition that determines use_fallback_kernel to avoid
66. `Used StrReplaceFile (sglang/python/sglang/srt/...ayers/rotary_embedding.py)`
67. Now let me run the test harness to verify the fix works.
68. Now let me run the test harness to verify the fix:
69. `Used Shell (/opt/venv/bin/python3 /wo...pace/test_harness.py 2>&1)`
70. The error changed! Now it's not a CUDA JIT compilation error, but a device mis
71. The error changed from CUDA JIT to a device mismatch issue. The cos_sin_cache
72. `Used ReadFile (test_harness.py)`
73. I see the issue. The test creates the RotaryEmbedding module on CPU (default),
74. The issue is that forward_native doesn't handle device placement like forward_
75. `Used ReadFile (sglang/python/sglang/srt/...ayers/rotary_embedding.py)`
76. I need to add device handling to forward_native. The cache needs to be moved t
77. `Used StrReplaceFile (sglang/python/sglang/srt/...ayers/rotary_embedding.py)`
78. Now let me run the test again to see if the fix works.
79. `Used Shell (/opt/venv/bin/python3 /wo...pace/test_harness.py 2>&1)`
80. The test passes now with a score of 100.0. Let me update the optimization_stat
81. The fix works. Now let me update the optimization state file:
82. `Used WriteFile (optimization_state.json)`
83. Let me verify the fix one more time by running the test harness to confirm the
84. Let me verify the fix one more time:
85. `Used Shell (/opt/venv/bin/python3 /wo...pace/test_harness.py 2>&1)`
86. The fix is verified. The test passes with a score of 100.0. Let me provide a s
87. The fix is complete and verified. Here's a summary of the changes made:
88. The task has been completed successfully. The test harness now passes with SCO
89. `Used Shell (/opt/venv/bin/python3 /wo...pace/test_harness.py 2>&1)`
90. `Used ReadFile (optimization_state.json)`
91. The test passes with SCORE: 100.0 and the optimization state file is properly
92. Adding forward_hip() method that uses forward_native() instead
93. Fixed use_fallback_kernel condition to be CUDA-specific
94. Added proper device handling in forward_native()
95. The task is complete. The test harness passes with SCORE: 100.0.
96. Modified /workspace/sglang/python/sglang/srt/layers/rotary_embedding.py
97. Added forward_hip() method that uses forward_native() instead of falling thr
98. Fixed use_fallback_kernel condition to only apply to CUDA devices
99. Added proper device handling in forward_native() for cos_sin_cache
100. **Agent finished** (STOP)

---
*Generated by amdpilot orchestrator*