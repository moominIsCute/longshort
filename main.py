import os
import re
import json
import time
import tempfile
import subprocess
import sys
import argparse
import importlib
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("오류: .env 파일에 GEMINI_API_KEY를 설정해주세요.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)
MODEL = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

AVAILABLE_PROMPTS = [
    p.stem for p in Path("prompts").glob("v*.py") if p.stem != "__init__"
]


def load_prompt(version: str) -> str:
    try:
        module = importlib.import_module(f"prompts.{version}")
        return module.PROMPT
    except ModuleNotFoundError:
        print(f"오류: 프롬프트 버전 '{version}'을 찾을 수 없습니다.")
        print(f"사용 가능한 버전: {', '.join(sorted(AVAILABLE_PROMPTS))}")
        sys.exit(1)


def upload_video(video_path: str):
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {video_path}")

    print(f"영상 업로드 중: {path.name}")
    video_file = client.files.upload(file=path, config={"display_name": "video"})
    print(f"업로드 완료. 처리 대기 중...")

    while video_file.state == "PROCESSING":
        time.sleep(3)
        video_file = client.files.get(name=video_file.name)
        print(".", end="", flush=True)

    print()

    if video_file.state == "FAILED":
        raise RuntimeError("영상 처리 실패. 다른 파일을 시도해주세요.")

    print(f"영상 준비 완료: {video_file.display_name}")
    return video_file


def mmss_to_seconds(ts: float) -> float:
    """11.07 형식(분.초)을 순수 초(667.0)로 변환."""
    minutes = int(ts)
    centiseconds = round((ts - minutes) * 100)
    if 0 <= centiseconds <= 59:
        return minutes * 60 + centiseconds
    return ts


def parse_clips(text: str) -> list[dict]:
    match = re.search(r"```clip_json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        return []
    clips = json.loads(match.group(1)).get("clips", [])

    # 총 클립 길이가 비정상적으로 짧으면 MM.SS 형식으로 판단해 변환
    total = sum(c["end"] - c["start"] for c in clips)
    if clips and total < 5:
        clips = [{"start": mmss_to_seconds(c["start"]), "end": mmss_to_seconds(c["end"])} for c in clips]

    return clips


def encode_clip(video_path: str, start: float, end: float, output_path: str):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", video_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "fast",
            output_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_video(video_path: str, clips: list[dict], output_path: str):
    with tempfile.TemporaryDirectory() as tmp:
        segment_files = []
        for i, clip in enumerate(clips):
            seg = os.path.join(tmp, f"seg_{i:03d}.mp4")
            encode_clip(video_path, clip["start"], clip["end"], seg)
            segment_files.append(seg)

        concat_list = os.path.join(tmp, "list.txt")
        with open(concat_list, "w") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def build_individual_clips(video_path: str, clips: list[dict], base: Path):
    output_paths = []
    for i, clip in enumerate(clips, start=1):
        output_path = base.with_stem(base.stem + f"_{i:02d}").with_suffix(".mp4")
        counter = 1
        while output_path.exists():
            output_path = base.with_stem(base.stem + f"_{i:02d} ({counter})").with_suffix(".mp4")
            counter += 1
        encode_clip(video_path, clip["start"], clip["end"], str(output_path))
        output_paths.append(output_path)
    return output_paths


def ask_about_video(video_file, question: str, system_prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=[video_file, question],
        config={"system_instruction": system_prompt},
    )
    return response.text


def main():
    parser = argparse.ArgumentParser(description="🎬 쇼츠크리에이터 - 영상 분석기")
    parser.add_argument("video_path", nargs="?", help="분석할 영상 파일 경로")
    parser.add_argument(
        "--prompt",
        default="v1",
        help=f"사용할 프롬프트 버전 (기본값: v1 / 사용 가능: {', '.join(sorted(AVAILABLE_PROMPTS))})",
    )
    args = parser.parse_args()

    if args.video_path:
        video_path = args.video_path
        prompt_version = args.prompt
    else:
        print("🎬 쇼츠크리에이터 - 영상 분석기")
        print(f"프롬프트 버전을 선택하세요 ({', '.join(sorted(AVAILABLE_PROMPTS))})")
        prompt_version = input("버전: ").strip()
        if prompt_version not in AVAILABLE_PROMPTS:
            print(f"오류: '{prompt_version}'은 없는 버전입니다. 사용 가능: {', '.join(sorted(AVAILABLE_PROMPTS))}")
            sys.exit(1)

        print("분석할 영상 파일 경로를 입력하세요.")
        video_path = input("영상 경로: ").strip()
        if not video_path:
            print("파일 경로를 입력해야 합니다.")
            sys.exit(1)

    system_prompt = load_prompt(prompt_version)
    print(f"프롬프트 버전: {prompt_version}")

    total_start = time.time()

    try:
        video_file = upload_video(video_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"오류: {e}")
        sys.exit(1)

    print("\n분석 중...\n")
    api_start = time.time()
    try:
        result = ask_about_video(video_file, "이 영상을 멀티모달 분석 모드로 전체 분석해줘.", system_prompt)
        api_end = time.time()
        print(result)

        clips = parse_clips(result)
        if clips:
            base = Path(video_path).with_stem(Path(video_path).stem + "_shorts")
            print(f"\n편집점 {len(clips)}개 감지. 영상 생성 중...")
            try:
                if prompt_version == "v1":
                    outputs = build_individual_clips(video_path, clips, base)
                    for p in outputs:
                        print(f"  저장: {p.name}")
                    print(f"완료: {len(outputs)}개 영상 생성")
                else:
                    output_path = base.with_suffix(".mp4")
                    counter = 1
                    while output_path.exists():
                        output_path = base.with_stem(base.stem + f" ({counter})").with_suffix(".mp4")
                        counter += 1
                    build_video(video_path, clips, str(output_path))
                    print(f"완료: {output_path.name}")
            except subprocess.CalledProcessError:
                print("오류: ffmpeg 실행 실패. ffmpeg이 설치되어 있는지 확인해주세요.")
        else:
            print("\n편집점 JSON을 찾지 못했습니다. 영상 생성을 건너뜁니다.")
    except Exception as e:
        api_end = time.time()
        print(f"오류 발생: {e}")

    total_elapsed = time.time() - total_start
    api_elapsed = api_end - api_start
    print(f"\n[시간] API 호출: {api_elapsed:.1f}초 | 전체 소요: {total_elapsed:.1f}초")


if __name__ == "__main__":
    main()
