"""
API 테스트 스크립트

사용법:
    python test_api.py --image <이미지 경로> --url http://localhost:8000
"""

import requests
import argparse
import json
from pathlib import Path


def test_health(base_url):
    """헬스 체크 테스트"""
    print("\n" + "=" * 60)
    print("🏥 헬스 체크 테스트")
    print("=" * 60)

    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()

        print(f"✅ 상태: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return True

    except Exception as e:
        print(f"❌ 오류: {e}")
        return False


def test_analyze(base_url, image_path):
    """이미지 분석 테스트"""
    print("\n" + "=" * 60)
    print("📸 이미지 분석 테스트")
    print("=" * 60)

    image_path = Path(image_path)

    if not image_path.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {image_path}")
        return False

    print(f"📁 파일: {image_path.name} ({image_path.stat().st_size / 1024:.1f} KB)")

    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/jpeg")}

            print(f"🚀 요청 중... {base_url}/analyze")
            response = requests.post(f"{base_url}/analyze", files=files, timeout=30)

        response.raise_for_status()

        result = response.json()

        print(f"\n✅ 상태: {response.status_code}")
        print("\n📋 분석 결과:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # 결과 해석
        if result.get("status") == "success":
            data = result.get("data", {})
            print("\n📊 요약:")
            print(f"  진단: {data.get('diagnosis')}")
            print(f"  신뢰도: {data.get('confidence')}")
            print(f"  PLA: {data.get('pla_cm2')} cm²")
            print(f"  초록색 픽셀: {data.get('green_pixels')}")

        return True

    except requests.exceptions.Timeout:
        print("❌ 타임아웃: 요청 시간 초과 (30초)")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ 연결 오류: 서버에 연결할 수 없습니다")
        print(f"   확인: {base_url} 서버가 실행 중인지 확인하세요")
        return False
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Basil Analyzer API 테스트 스크립트"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="테스트할 이미지 경로",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API 서버 URL (기본값: http://localhost:8000)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🧪 Basil Health Analyzer API 테스트")
    print("=" * 60)
    print(f"API URL: {args.url}")

    # 헬스 체크
    if not test_health(args.url):
        print("\n⚠️ 서버 연결 실패. 서버가 실행 중인지 확인하세요.")
        print(f"   실행 명령어: cd my_ai_service && uvicorn app.main:app --reload")
        return

    # 이미지 분석
    test_analyze(args.url, args.image)

    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
