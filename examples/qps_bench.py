"""QPS 压测 — 逐步增加并发找出模型的限流阈值。

用法:
  python examples/qps_bench.py --api-key sk-xxx --base-url https://... --model qwen-plus
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def single_call(provider, model: str, idx: int) -> tuple[int, bool, str]:
    """单次 API 调用。返回 (index, success, error_msg)。"""
    try:
        await provider.create_message(
            model=model,
            system_prompt="Reply with just 'OK'.",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10,
        )
        return idx, True, ""
    except Exception as e:
        return idx, False, str(e)


async def test_concurrency(provider, model: str, concurrency: int,
                           rounds: int = 3) -> tuple[int, int, int]:
    """返回 (success, rate_limited, other_errors)。"""
    total_ok = 0
    total_429 = 0
    total_other = 0

    for r in range(rounds):
        tasks = [single_call(provider, model, i) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)

        for _, ok, err in results:
            if ok:
                total_ok += 1
            elif "429" in err or "RateLimit" in err or "rate" in err.lower():
                total_429 += 1
            else:
                total_other += 1
                if total_other <= 2:  # 只打印前两条避免刷屏
                    print(f"    [ERROR] {err[:200]}")

        if r < rounds - 1:
            await asyncio.sleep(1.0)

    return total_ok, total_429, total_other


async def main():
    parser = argparse.ArgumentParser(description="QPS Benchmark for LLM API")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-test", type=int, default=10,
                       help="最大测试并发数")
    args = parser.parse_args()

    from agent_team import OpenAIProvider

    print(f"QPS Benchmark — {args.model}")
    print(f"Endpoint: {args.base_url}")
    print(f"Testing concurrency 1..{args.max_test}")
    print("-" * 50)

    best = 0
    for c in range(1, args.max_test + 1):
        # 每次测试用新 provider（避免信号量干扰）
        provider = OpenAIProvider(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            max_concurrent=c,  # 允许全部并发发出
        )

        t0 = time.perf_counter()
        ok, rate_limited, other_err = await test_concurrency(provider, args.model, c, rounds=3)
        elapsed = time.perf_counter() - t0

        total = ok + rate_limited + other_err
        status = "CLEAN" if rate_limited == 0 else f"RATE-LIMITED ({rate_limited})"
        if other_err > 0:
            status += f" OTHER-ERR ({other_err})"
        print(f"  concurrency={c:2d}  |  {ok}/{total} ok  |  {elapsed:.1f}s  |  {status}")

        if other_err > 0 and ok == 0:
            print(f"\n  All requests failing. Check API key, base_url, or model name.")
            break

        if rate_limited == 0 and other_err == 0:
            best = c
        elif rate_limited > 0:
            print(f"\n  Hit rate limit at concurrency={c}. Safe max: {best}")
            break
    else:
        print(f"\n  No rate limit detected up to {args.max_test}. Safe: >= {args.max_test}")

    if best > 0:
        print(f"\n  Recommended --max-concurrent: {best}")


if __name__ == "__main__":
    asyncio.run(main())
