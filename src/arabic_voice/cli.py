from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from arabic_voice.base import BackendUnavailableError, MissingDependencyError
from arabic_voice.paths import OUTPUTS_DIR, SCRIPTS_DIR, ensure_dirs
from arabic_voice.registry import get_backend, list_backend_ids, list_backends
from arabic_voice.types import SynthesisRequest

app = typer.Typer(
    name="arabic-voice",
    help="Emirati Arabic TTS multi-backend wrappers (Mac-friendly).",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command("list")
def list_cmd() -> None:
    """List available TTS backends and license/Mac status."""
    table = Table(title="Emirati TTS Backends")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("License")
    table.add_column("Mac")
    table.add_column("Commercial")
    table.add_column("Extra")
    table.add_column("Ref?")

    for backend in list_backends():
        info = backend.info()
        table.add_row(
            info.id,
            info.name,
            info.license,
            info.mac_status,
            "yes" if info.commercial_ok else "NO (NC)",
            info.extra or "-",
            "yes" if info.requires_ref_audio else "no",
        )
    console.print(table)


@app.command("synth")
def synth_cmd(
    backend: str = typer.Option(..., "--backend", "-b", help="Backend id"),
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Text to synthesize"),
    text_file: Optional[Path] = typer.Option(
        None, "--text-file", help="UTF-8 file with text to synthesize"
    ),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output WAV path"),
    ref_audio: Optional[Path] = typer.Option(None, "--ref-audio", help="Reference WAV for cloning backends"),
    ref_text: Optional[str] = typer.Option(None, "--ref-text", help="Transcript of reference audio"),
    speaker: Optional[str] = typer.Option(None, "--speaker", help="Speaker id if applicable"),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed (Piper supports this)"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Optional RNG seed"),
) -> None:
    """Synthesize speech with one backend."""
    ensure_dirs()
    if not text and not text_file:
        raise typer.BadParameter("Provide --text or --text-file")
    if text_file:
        text = text_file.read_text(encoding="utf-8")
    assert text is not None

    if out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUTS_DIR / f"{backend}_{stamp}.wav"

    req = SynthesisRequest(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        speaker=speaker,
        speed=speed,
        seed=seed,
    )

    b = get_backend(backend)
    console.print(f"[bold]Loading[/bold] {backend} …")
    try:
        b.ensure_loaded()
        result = b.synthesize(req)
    except (MissingDependencyError, BackendUnavailableError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    path = result.write_wav(out)
    console.print(f"[green]Wrote[/green] {path}  ({result.sample_rate} Hz, {len(result.audio)} samples)")
    if result.meta:
        console.print(f"[dim]meta: {result.meta}[/dim]")


@app.command("compare")
def compare_cmd(
    text: Optional[str] = typer.Option(None, "--text", "-t"),
    text_file: Optional[Path] = typer.Option(None, "--text-file"),
    backends: str = typer.Option(
        "piper_emirati,audar_flash,habibi_uae",
        "--backends",
        help="Comma-separated backend ids",
    ),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir"),
    ref_audio: Optional[Path] = typer.Option(None, "--ref-audio"),
    ref_text: Optional[str] = typer.Option(None, "--ref-text"),
    continue_on_error: bool = typer.Option(
        True, "--continue-on-error/--fail-fast", help="Keep going if one backend fails"
    ),
) -> None:
    """Run the same text through multiple backends and write WAVs side-by-side."""
    ensure_dirs()
    if not text and not text_file:
        # Default prompt file
        default_prompts = SCRIPTS_DIR / "prompts_emirati.txt"
        if default_prompts.exists():
            lines = [
                ln.strip()
                for ln in default_prompts.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            text = lines[0] if lines else None
        if not text:
            raise typer.BadParameter("Provide --text, --text-file, or prompts_emirati.txt")
    if text_file:
        text = text_file.read_text(encoding="utf-8").strip()
    assert text is not None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = out_dir or (OUTPUTS_DIR / f"compare_{stamp}")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "prompt.txt").write_text(text, encoding="utf-8")

    ids = [b.strip() for b in backends.split(",") if b.strip()]
    unknown = [i for i in ids if i not in list_backend_ids()]
    if unknown:
        console.print(f"[red]Unknown backends: {unknown}[/red]")
        raise typer.Exit(1)

    summary: list[tuple[str, str]] = []
    for bid in ids:
        console.print(f"\n[bold cyan]=== {bid} ===[/bold cyan]")
        b = get_backend(bid, reload=True)
        req = SynthesisRequest(text=text, ref_audio=ref_audio, ref_text=ref_text)
        try:
            b.ensure_loaded()
            result = b.synthesize(req)
            path = result.write_wav(dest / f"{bid}.wav")
            console.print(f"[green]OK[/green] {path}")
            summary.append((bid, str(path)))
        except Exception as exc:
            console.print(f"[red]FAIL[/red] {bid}: {exc}")
            summary.append((bid, f"ERROR: {exc}"))
            if not continue_on_error:
                raise typer.Exit(1) from exc

    console.print(f"\n[bold]Results in[/bold] {dest}")
    for bid, status in summary:
        console.print(f"  • {bid}: {status}")


@app.command("compare-prompts")
def compare_prompts_cmd(
    prompts_file: Path = typer.Option(
        SCRIPTS_DIR / "prompts_emirati.txt",
        "--prompts",
        help="One prompt per line (# comments ok)",
    ),
    backends: str = typer.Option(
        "piper_emirati,audar_flash,habibi_uae",
        "--backends",
    ),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir"),
    ref_audio: Optional[Path] = typer.Option(None, "--ref-audio"),
    ref_text: Optional[str] = typer.Option(None, "--ref-text"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only first N prompts"),
) -> None:
    """Synthesize every prompt in a file with each backend."""
    ensure_dirs()
    if not prompts_file.exists():
        console.print(f"[red]Missing prompts file: {prompts_file}[/red]")
        raise typer.Exit(1)

    prompts = [
        ln.strip()
        for ln in prompts_file.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if limit is not None:
        prompts = prompts[:limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = out_dir or (OUTPUTS_DIR / f"prompts_{stamp}")
    dest.mkdir(parents=True, exist_ok=True)

    ids = [b.strip() for b in backends.split(",") if b.strip()]
    for i, prompt in enumerate(prompts, start=1):
        prompt_dir = dest / f"{i:02d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        console.print(f"\n[bold]Prompt {i}/{len(prompts)}[/bold]: {prompt[:60]}…")
        for bid in ids:
            try:
                b = get_backend(bid)
                b.ensure_loaded()
                result = b.synthesize(
                    SynthesisRequest(text=prompt, ref_audio=ref_audio, ref_text=ref_text)
                )
                result.write_wav(prompt_dir / f"{bid}.wav")
                console.print(f"  [green]{bid}[/green]")
            except Exception as exc:
                console.print(f"  [red]{bid}[/red]: {exc}")

    console.print(f"\n[bold]Wrote[/bold] {dest}")


if __name__ == "__main__":
    app()
