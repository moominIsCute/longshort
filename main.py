import os
import time
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

    try:
        video_file = upload_video(video_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"오류: {e}")
        sys.exit(1)

    print("\n분석 중...\n")
    try:
        result = ask_about_video(video_file, "이 영상을 멀티모달 분석 모드로 전체 분석해줘.", system_prompt)
        print(result)
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
