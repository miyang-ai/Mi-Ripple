"""Optional cleaned-reference regeneration through a pluggable client."""

from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path
from typing import Protocol

import httpx
from PIL import Image

from .common import image_size, now_iso, sha256_of, write_json

DEFAULT_BASE_URL = "https://miyang.cn/api/v1"
DEFAULT_MODEL = "miyang/gpt-image-2-internal"
MODEL_4K = "openai_kr/miyang/gpt-image-2-4k-internal"
MAX_SIDE = 3840

PROMPT_TEMPLATE = {
    "ko": (
        "이 참고 일러스트를 한 번의 깨끗한 렌더링으로 다시 그려 주세요. 구도, 프레이밍, "
        "인물의 정체성, 포즈, 표정, 머리 모양, 복장, 색, 빛, 화풍은 참고와 정확히 같게 "
        "유지해 주세요. 머리카락은 연속된 가닥으로, 넓은 면은 매끈한 그라데이션으로 "
        "표현하고 반복되는 비늘, 벌집, 입자 또는 주기적 미세 패턴을 넣지 마세요. "
        "어떤 사물도 추가하거나 빼지 마세요.{hint}"
    ),
    "en": (
        "Redraw this reference illustration as one clean rendering pass. Keep exactly the "
        "same composition, framing, character identity, pose, expression, hairstyle, "
        "clothing, colors, lighting, and painting style. Render hair as continuous flowing "
        "strands and broad surfaces as smooth gradients. Do not introduce overlapping "
        "scale-like patches, honeycomb texture, granules, noise, or periodic micro-patterns. "
        "Do not add or remove any object.{hint}"
    ),
    "zh": (
        "把这张参考插画重新画成一次干净的渲染。构图、取景、人物身份、姿势、表情、发型、"
        "服装、颜色、光线和画风都与参考一致。头发使用连续流畅的发丝，大面积使用平滑渐变；"
        "不要引入叠瓦状鳞片、蜂窝纹理、颗粒、噪点或周期性微图案。不要增删任何物体。{hint}"
    ),
}


def build_prompt(hint: dict[str, str] | None = None) -> dict[str, str]:
    hint = hint or {}
    return {
        language: template.format(
            hint=(" " + hint[language]) if hint.get(language) else ""
        )
        for language, template in PROMPT_TEMPLATE.items()
    }


def request_size(width: int, height: int) -> str:
    scale = min(1.0, MAX_SIDE / max(width, height))
    requested_width = max(16, int(width * scale) // 16 * 16)
    requested_height = max(16, int(height * scale) // 16 * 16)
    return f"{requested_width}x{requested_height}"


class RegenClient(Protocol):
    def edit(
        self,
        image_bytes: bytes,
        filename: str,
        prompt: str,
        model: str,
        size: str,
        quality: str,
    ) -> bytes: ...


class MiyangClient:
    """MIYANG image-edit client with bounded, explicit network behavior."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        read_timeout_seconds: float | None = None,
        fallback_proxy: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.read_timeout_seconds = read_timeout_seconds
        self.fallback_proxy = fallback_proxy
        self.last_meta: dict = {}

    def _post(
        self,
        files: dict,
        data: dict,
        timeout: httpx.Timeout,
        proxy: str | None,
    ) -> httpx.Response:
        with httpx.Client(
            timeout=timeout,
            proxy=proxy,
            trust_env=False,
        ) as client:
            return client.post(
                f"{self.base_url}/images/edits",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                data=data,
            )

    def edit(
        self,
        image_bytes: bytes,
        filename: str,
        prompt: str,
        model: str,
        size: str,
        quality: str,
    ) -> bytes:
        read_timeout = self.read_timeout_seconds or (
            600.0 if model == MODEL_4K else 300.0
        )
        timeout = httpx.Timeout(
            connect=30.0,
            read=read_timeout,
            write=120.0,
            pool=30.0,
        )
        files = {"image": (filename, image_bytes, "image/png")}
        data = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "quality": quality,
        }
        try:
            response = self._post(files, data, timeout, None)
            self.last_meta["proxy"] = False
        except (httpx.ConnectError, httpx.ConnectTimeout) as error:
            if not self.fallback_proxy:
                raise
            self.last_meta["direct_error"] = repr(error)
            response = self._post(
                files,
                data,
                timeout,
                self.fallback_proxy,
            )
            self.last_meta["proxy"] = True
        if response.status_code == 402:
            raise RuntimeError("MIYANG returned 402: insufficient balance")
        if response.status_code == 429:
            raise RuntimeError("MIYANG returned 429: rate limit exceeded")
        if response.status_code != 200:
            raise RuntimeError(
                f"MIYANG returned {response.status_code}: "
                f"{response.text[:400]}"
            )
        item = response.json()["data"][0]
        self.last_meta["response_keys"] = sorted(item.keys())
        if item.get("b64_json"):
            return base64.b64decode(item["b64_json"])
        url = item["url"]
        with httpx.Client(
            timeout=httpx.Timeout(300.0),
            proxy=self.fallback_proxy if self.last_meta.get("proxy") else None,
            trust_env=False,
        ) as client:
            cached = client.get(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            cached.raise_for_status()
            return cached.content


def regenerate(
    reference: Path,
    out_path: Path,
    client: RegenClient,
    model: str = DEFAULT_MODEL,
    quality: str = "high",
    hint: dict[str, str] | None = None,
    prompt_lang: str = "en",
    reference_role: str = (
        "single reference: composition + character + palette + style"
    ),
    extra_sidecar: dict | None = None,
    size: str | None = None,
    prompt_override: dict[str, str] | None = None,
) -> dict:
    prompts = prompt_override or build_prompt(hint)
    if set(prompts) != {"ko", "en", "zh"}:
        raise ValueError("prompt_override must contain ko, en, and zh")
    source_width, source_height = image_size(reference)
    requested_size = size or request_size(source_width, source_height)
    target_width, target_height = (
        int(value) for value in requested_size.split("x")
    )
    target_ratio = (
        target_width / target_height
        if size is not None
        else source_width / source_height
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path = out_path.with_suffix(".prompt.json")
    write_json(
        prompt_path,
        {
            "purpose": "cleaned-reference faithful regeneration",
            "model": model,
            "size": requested_size,
            "quality": quality,
            "prompt_lang_sent": prompt_lang,
            "reference": str(reference),
            "reference_sha256": sha256_of(reference),
            "reference_role": reference_role,
            "prompt_ko": prompts["ko"],
            "prompt_en": prompts["en"],
            "prompt_zh": prompts["zh"],
            "hint": hint or {},
            "time": now_iso(),
        },
    )
    requested_at = now_iso()
    raw = client.edit(
        reference.read_bytes(),
        reference.name,
        prompts[prompt_lang],
        model,
        requested_size,
        quality,
    )
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    actual_size = image.size
    cropped = False
    if abs(actual_size[0] / actual_size[1] - target_ratio) > 0.01:
        if actual_size[0] / actual_size[1] > target_ratio:
            new_width = int(round(actual_size[1] * target_ratio))
            x_offset = (actual_size[0] - new_width) // 2
            image = image.crop(
                (x_offset, 0, x_offset + new_width, actual_size[1])
            )
        else:
            new_height = int(round(actual_size[0] / target_ratio))
            y_offset = (actual_size[1] - new_height) // 2
            image = image.crop(
                (0, y_offset, actual_size[0], y_offset + new_height)
            )
        cropped = True
    image.save(out_path)
    saved_size = image_size(out_path)
    endpoint = getattr(client, "base_url", DEFAULT_BASE_URL).rstrip("/")
    sidecar = {
        "stage": "regenerate",
        "grade": "candidate_not_deliverable_until_human_accepts",
        "provider": type(client).__name__,
        "model": model,
        "endpoint": f"{endpoint}/images/edits",
        "requested_size": requested_size,
        "actual_size_from_api": list(actual_size),
        "saved_size": list(saved_size),
        "size_verified": saved_size == (target_width, target_height),
        "center_cropped_to_requested_ratio": cropped,
        "quality": quality,
        "prompt_file": str(prompt_path),
        "prompt_lang_sent": prompt_lang,
        "references": [
            {
                "path": str(reference),
                "role": reference_role,
                "sha256": sha256_of(reference),
            }
        ],
        "output": str(out_path),
        "output_sha256": sha256_of(out_path),
        "requested_at": requested_at,
        "saved_at": now_iso(),
        "client_meta": getattr(client, "last_meta", {}),
        **(extra_sidecar or {}),
    }
    write_json(out_path.with_suffix(".json"), sidecar)
    return sidecar


def client_from_env(
    read_timeout_seconds: float | None = None,
) -> MiyangClient:
    api_key = os.environ.get("MIYANG_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "MIYANG_API_KEY is required for optional regeneration"
        )
    return MiyangClient(
        api_key,
        base_url=os.environ.get("MIYANG_BASE_URL", DEFAULT_BASE_URL),
        read_timeout_seconds=read_timeout_seconds,
        fallback_proxy=os.environ.get("MIYANG_PROXY") or None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate once from a cleaned reference using MIYANG."
    )
    parser.add_argument("reference")
    parser.add_argument("out_path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quality", default="high")
    parser.add_argument("--lang", default="en", choices=("ko", "en", "zh"))
    arguments = parser.parse_args()
    sidecar = regenerate(
        Path(arguments.reference),
        Path(arguments.out_path),
        client_from_env(),
        arguments.model,
        arguments.quality,
        prompt_lang=arguments.lang,
    )
    print(
        {
            "output": sidecar["output"],
            "saved_size": sidecar["saved_size"],
            "size_verified": sidecar["size_verified"],
        }
    )


if __name__ == "__main__":
    main()
