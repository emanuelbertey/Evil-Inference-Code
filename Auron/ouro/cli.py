"""CLI entry point for Ouro model inference."""

import argparse
import os

from dotenv import load_dotenv


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate text from Auron (Chimera) language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  ouro "The history of"
  ouro "def fibonacci(n):" --model nyxia/Auron-510M
  ouro "Hello world" --temp 1.0 --max-tokens 200
  ouro "Scientists have" --temp 0.7 --presence-pen 1.5 --rep-pen 1.0
""",
    )
    parser.add_argument("prompt", type=str, help="Text prompt")
    parser.add_argument(
        "--model", type=str, default="nyxia/Auron-279M",
        help="HuggingFace repo ID (default: nyxia/Auron-279M). Available: nyxia/Auron-279M, nyxia/Auron-510M, nyxia/Auron-1.1B",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--rep-pen", type=float, default=1.0)
    parser.add_argument("--presence-pen", type=float, default=1.5)
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, auto")
    parser.add_argument("--token", type=str, default=None, help="HF token (or set HF_TOKEN in .env)")
    args = parser.parse_args()

    from .generate import load_model, generate

    token = args.token or os.environ.get("HF_TOKEN")
    model, tokenizer, device = load_model(args.model, device=args.device, token=token)

    generate(
        model, tokenizer, device, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        rep_pen=args.rep_pen,
        presence_pen=args.presence_pen,
    )
